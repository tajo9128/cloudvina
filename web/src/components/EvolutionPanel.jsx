import React, { useEffect, useRef, useState } from 'react';
import * as $3Dmol from '3dmol/build/3Dmol.js';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { API_URL } from '../config';

const EvolutionPanel = ({ jobId, initialPdbqt, onClose }) => {
    const viewerRef = useRef(null);
    const [viewer, setViewer] = useState(null);
    const [socket, setSocket] = useState(null);
    const [evolutionData, setEvolutionData] = useState([]);
    const [status, setStatus] = useState('Connecting...');
    const [currentGen, setCurrentGen] = useState(0);
    const [bestScore, setBestScore] = useState(null);
    const [isPaused, setIsPaused] = useState(false);

    // Initialize 3Dmol Viewer
    useEffect(() => {
        if (!viewerRef.current) return;

        const v = new $3Dmol.GLViewer(viewerRef.current, {
            backgroundColor: 'white',
            id: 'evolution-viewer'
        });
        setViewer(v);

        // Load initial molecule (The "Original")
        if (initialPdbqt) {
            // Add original as Red "Ghost"
            v.addModel(initialPdbqt, "pdbqt");
            v.setStyle({ model: 0 }, { stick: { color: "red", opacity: 0.5 } });
            v.zoomTo();
            v.render();
        }

        return () => {
            // Cleanup
        };
    }, []);

    // ... (WebSocket logic) ...

    // Update Viewer if SDF data is present
    if (data.sdf && viewer) {
        // Keep model 0 (Original), remove others (previous generations)
        const models = viewer.models;
        if (models.length > 1) {
            viewer.removeModel(models[models.length - 1]);
        }

        // Add new evolved model as Green
        viewer.addModel(data.sdf, "sdf");
        // Last model is the new one
        viewer.setStyle({ model: -1 }, { stick: { colorscheme: "greenCarbon" } });

        viewer.render();
    }

    // Connect to WebSocket
    useEffect(() => {
        if (!jobId) return;

        const wsUrl = API_URL.replace('http', 'ws') + `/ws/evolve/${jobId}`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            setStatus('Connected. Starting Evolution...');
            // Send initial config if needed
            ws.send(JSON.stringify({ action: 'start' }));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.status === 'Completed') {
                setStatus('Evolution Completed');
                ws.close();
                return;
            }

            if (data.gen) {
                setCurrentGen(data.gen);
                setBestScore(data.score);
                setEvolutionData(prev => [...prev, { gen: data.gen, score: data.score }]);
                setStatus(`Evolving... Gen ${data.gen}`);

                // Update Viewer if SDF data is present
                if (data.sdf && viewer) {
                    viewer.removeAllModels();
                    viewer.addModel(data.sdf, "sdf");
                    viewer.setStyle({}, { stick: { colorscheme: "cyanCarbon" } });
                    viewer.zoomTo();
                    viewer.render();
                }
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            setStatus('Connection Error');
        };

        ws.onclose = () => {
            if (status !== 'Evolution Completed') {
                setStatus('Disconnected');
            }
        };

        setSocket(ws);

        return () => {
            if (ws) ws.close();
        };
    }, [jobId, viewer]);

    const handleStop = () => {
        if (socket) {
            socket.close();
            setStatus('Stopped by User');
        }
    };

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-6xl h-[80vh] flex flex-col overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-slate-200 flex justify-between items-center bg-slate-50">
                    <div>
                        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                            <span className="text-purple-600">✨</span> AI Evolution
                        </h2>
                        <p className="text-sm text-slate-500">Refining ligand for better affinity</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="text-right">
                            <div className="text-xs text-slate-500 uppercase font-bold">Current Score</div>
                            <div className="text-2xl font-bold text-green-600">{bestScore ? bestScore.toFixed(2) : '-'}</div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-slate-200 rounded-full transition-colors"
                        >
                            <svg className="w-6 h-6 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
                    {/* Left: 3D Viewer */}
                    <div className="w-full lg:w-2/3 relative bg-slate-100">
                        <div ref={viewerRef} className="w-full h-full" />
                        <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur px-4 py-2 rounded-lg shadow-sm border border-slate-200">
                            <div className="text-xs font-bold text-slate-500 uppercase">Status</div>
                            <div className="font-medium text-slate-900 flex items-center gap-2">
                                {status === 'Evolving...' && (
                                    <span className="relative flex h-2 w-2">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                                    </span>
                                )}
                                {status}
                            </div>
                        </div>
                    </div>

                    {/* Right: Graph & Stats */}
                    <div className="w-full lg:w-1/3 border-l border-slate-200 bg-white flex flex-col">
                        <div className="p-6 flex-1">
                            <h3 className="text-sm font-bold text-slate-900 mb-4">Improvement Trajectory</h3>
                            <div className="h-64 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={evolutionData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis dataKey="gen" stroke="#94a3b8" fontSize={12} />
                                        <YAxis stroke="#94a3b8" fontSize={12} domain={['auto', 'auto']} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0' }}
                                            labelStyle={{ color: '#64748b' }}
                                        />
                                        <Line
                                            type="monotone"
                                            dataKey="score"
                                            stroke="#8b5cf6"
                                            strokeWidth={2}
                                            dot={false}
                                            activeDot={{ r: 6 }}
                                            animationDuration={300}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>

                            <div className="mt-6 space-y-4">
                                <div className="p-4 bg-purple-50 rounded-xl border border-purple-100">
                                    <div className="text-sm text-purple-800 font-medium mb-1">Generation</div>
                                    <div className="text-3xl font-bold text-purple-900">{currentGen}</div>
                                </div>
                            </div>
                        </div>

                        <div className="p-6 border-t border-slate-200 bg-slate-50">
                            <button
                                onClick={handleStop}
                                className="w-full py-3 px-4 bg-white border border-slate-300 text-slate-700 font-bold rounded-xl hover:bg-slate-50 transition-colors shadow-sm mb-3"
                            >
                                Stop & Save Current Result
                            </button>
                            <p className="text-xs text-center text-slate-500">
                                Evolution runs automatically for 50 generations or until convergence.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default EvolutionPanel;
