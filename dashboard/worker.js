/**
 * VM_FOOTBALL Training Dashboard - Cloudflare Worker
 * 
 * Receives training metrics from GCP VM via HTTP POST,
 * stores state in Durable Object, and pushes updates
 * to connected dashboard clients via WebSocket.
 * 
 * Deploy: npx wrangler deploy
 */

// ============================================================
// Durable Object: TrainingState
// ============================================================
export class TrainingState {
    constructor(state, env) {
        this.state = state;
        this.env = env;
        this.sessions = [];
        this.trainingData = {
            status: 'idle',
            experiments: {},
            lossData: {},
            mapData: {},
            logs: [],
        };

        // Load saved state
        this.state.blockConcurrencyWhile(async () => {
            const stored = await this.state.storage.get('trainingData');
            if (stored) this.trainingData = stored;
        });
    }

    async fetch(request) {
        const url = new URL(request.url);

        // WebSocket upgrade
        if (url.pathname === '/ws') {
            if (request.headers.get('Upgrade') !== 'websocket') {
                return new Response('Expected WebSocket', { status: 426 });
            }

            const pair = new WebSocketPair();
            const [client, server] = Object.values(pair);

            this.state.acceptWebSocket(server);
            this.sessions.push(server);

            // Send current state
            server.send(JSON.stringify({
                event: 'status',
                ...this.trainingData,
            }));

            return new Response(null, { status: 101, webSocket: client });
        }

        // HTTP API: receive update from training VM
        if (url.pathname === '/api/update' && request.method === 'POST') {
            const data = await request.json();
            await this.handleUpdate(data);
            return new Response(JSON.stringify({ ok: true }), {
                headers: { 'Content-Type': 'application/json' },
            });
        }

        // HTTP API: get current status
        if (url.pathname === '/api/status') {
            return new Response(JSON.stringify({
                event: 'status',
                ...this.trainingData,
            }), {
                headers: { 'Content-Type': 'application/json' },
            });
        }

        // HTTP API: reset state
        if (url.pathname === '/api/reset' && request.method === 'POST') {
            this.trainingData = {
                status: 'idle',
                experiments: {},
                lossData: {},
                mapData: {},
                logs: [],
            };
            await this.state.storage.put('trainingData', this.trainingData);
            this.broadcast({ event: 'status', ...this.trainingData });
            return new Response(JSON.stringify({ ok: true, message: 'State reset' }));
        }

        return new Response('Not found', { status: 404 });
    }

    async handleUpdate(data) {
        const { event, experiment } = data;

        switch (event) {
            case 'train_start':
                this.trainingData.status = 'training';
                this.trainingData.experiments[experiment] = {
                    model_type: data.model_type,
                    total_epochs: data.total_epochs,
                    config: data.config || {},
                    status: 'training',
                    startedAt: data.timestamp,
                };
                this.trainingData.lossData[experiment] = [];
                this.trainingData.mapData[experiment] = [];
                break;

            case 'epoch_end': {
                const metrics = data.metrics || {};
                const loss = metrics.train_loss || metrics.box_loss || metrics.loss || 0;
                const map50 = metrics['metrics/mAP50(B)'] || metrics.mAP50 || metrics.map50 || 0;

                if (!this.trainingData.lossData[experiment]) {
                    this.trainingData.lossData[experiment] = [];
                }
                this.trainingData.lossData[experiment].push({
                    epoch: data.epoch,
                    value: loss,
                });

                if (!this.trainingData.mapData[experiment]) {
                    this.trainingData.mapData[experiment] = [];
                }
                this.trainingData.mapData[experiment].push({
                    epoch: data.epoch,
                    value: map50,
                });

                if (this.trainingData.experiments[experiment]) {
                    this.trainingData.experiments[experiment].currentEpoch = data.epoch;
                    this.trainingData.experiments[experiment].lastMetrics = metrics;
                    this.trainingData.experiments[experiment].map50 = map50;
                }
                break;
            }

            case 'train_end':
                if (this.trainingData.experiments[experiment]) {
                    this.trainingData.experiments[experiment].status = 'complete';
                    this.trainingData.experiments[experiment].finalMetrics = data.final_metrics || {};
                    this.trainingData.experiments[experiment].totalTime = data.total_time_min;
                }

                // Check if all done
                const allDone = Object.values(this.trainingData.experiments)
                    .every(e => e.status === 'complete' || e.status === 'error');
                if (allDone) {
                    this.trainingData.status = 'complete';
                }
                break;

            case 'error':
                if (this.trainingData.experiments[experiment]) {
                    this.trainingData.experiments[experiment].status = 'error';
                    this.trainingData.experiments[experiment].error = data.error;
                }
                break;
        }

        // Add to log
        this.trainingData.logs.push({
            timestamp: data.timestamp || new Date().toISOString(),
            event,
            experiment,
            message: this.getLogMessage(data),
        });

        // Keep only last 200 logs
        if (this.trainingData.logs.length > 200) {
            this.trainingData.logs = this.trainingData.logs.slice(-200);
        }

        // Persist
        await this.state.storage.put('trainingData', this.trainingData);

        // Broadcast to all WebSocket clients
        this.broadcast(data);
    }

    getLogMessage(data) {
        switch (data.event) {
            case 'train_start':
                return `Started ${data.experiment} (${data.model_type})`;
            case 'epoch_end':
                return `${data.experiment} epoch ${data.epoch}/${data.total_epochs}`;
            case 'train_end':
                return `Completed ${data.experiment} in ${data.total_time_min}min`;
            case 'error':
                return `Error in ${data.experiment}: ${data.error}`;
            default:
                return JSON.stringify(data);
        }
    }

    broadcast(data) {
        const message = JSON.stringify(data);
        this.sessions = this.sessions.filter(ws => {
            try {
                ws.send(message);
                return true;
            } catch {
                return false;
            }
        });
    }

    async webSocketClose(ws) {
        this.sessions = this.sessions.filter(s => s !== ws);
    }

    async webSocketError(ws) {
        this.sessions = this.sessions.filter(s => s !== ws);
    }
}

// ============================================================
// Worker Entry Point
// ============================================================
export default {
    async fetch(request, env) {
        const url = new URL(request.url);

        // CORS headers
        const corsHeaders = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, X-API-Key',
        };

        // CORS preflight
        if (request.method === 'OPTIONS') {
            return new Response(null, { headers: corsHeaders });
        }

        // API key check for POST routes
        if (request.method === 'POST') {
            const apiKey = request.headers.get('X-API-Key');
            if (env.API_KEY && apiKey !== env.API_KEY) {
                return new Response(JSON.stringify({ error: 'Unauthorized' }), {
                    status: 401,
                    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
                });
            }
        }

        // Route to Durable Object for API/WS
        if (url.pathname.startsWith('/api/') || url.pathname === '/ws') {
            const id = env.TRAINING_STATE.idFromName('main');
            const obj = env.TRAINING_STATE.get(id);

            const resp = await obj.fetch(request);

            // Add CORS to response
            const newResp = new Response(resp.body, resp);
            Object.entries(corsHeaders).forEach(([k, v]) => {
                newResp.headers.set(k, v);
            });
            return newResp;
        }

        // Serve static dashboard (fallback)
        if (url.pathname === '/' || url.pathname === '/index.html') {
            return new Response('Dashboard is served via Cloudflare Pages', {
                headers: { 'Content-Type': 'text/plain' },
            });
        }

        return new Response('Not found', { status: 404, headers: corsHeaders });
    },
};
