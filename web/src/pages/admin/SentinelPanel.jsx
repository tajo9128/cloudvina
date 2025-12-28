import React, { useState } from 'react';
import { supabase } from '../../supabaseClient';

const Shield = () => <span>🛡️</span>;
const AlertTriangle = () => <span>⚠️</span>;
const CheckCircle = () => <span>✅</span>;
const Play = () => <span>▶️</span>;
const RefreshCw = () => <span>🔄</span>;

const SentinelPanel = () => {
    const [scanning, setScanning] = useState(false);
    const [report, setReport] = useState(null);
    const [history, setHistory] = useState([]);

    const runScan = async () => {
        setScanning(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            const response = await fetch(`${apiUrl}/admin/sentinel/scan`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${session?.access_token}` }
            });
            const result = await response.json();

            if (result.status === 'success') {
                setReport(result.report);
                setHistory(prev => [result.report, ...prev]);
            }
        } catch (error) {
            console.error("Sentinel Scan Failed:", error);
        } finally {
            setScanning(false);
        }
    };

    return (
        <div className="bg-slate-800/40 border border-primary-500/20 rounded-xl backdrop-blur-sm p-6 space-y-4">
            <div className="flex justify-between items-center border-b border-primary-500/10 pb-4">
                <div>
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <Shield /> BioDockify Sentinel
                    </h2>
                    <p className="text-slate-400 text-sm">Self-Healing Infrastructure Monitor</p>
                </div>
                <button
                    onClick={runScan}
                    disabled={scanning}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${scanning ? 'bg-slate-700 text-slate-500' : 'bg-primary-500 hover:bg-primary-600 text-white shadow-lg shadow-primary-500/20'}`}
                >
                    {scanning ? <RefreshCw className="animate-spin" /> : <Play />}
                    {scanning ? 'Scanning...' : 'Run Auto-Heal'}
                </button>
            </div>

            {/* Current Status */}
            {!report && !scanning && (
                <div className="text-center py-8 text-slate-500">
                    <Shield className="text-4xl opacity-20 mx-auto mb-2" />
                    <p>System is monitoring securely.</p>
                </div>
            )}

            {/* Scan Report */}
            {report && (
                <div className="space-y-4 animate-in fade-in slide-in-from-top-4 duration-500">
                    <div className="grid grid-cols-2 gap-4">
                        <div className={`p-4 rounded-lg border ${report.anomalies_detected > 0 ? 'bg-red-500/10 border-red-500/30' : 'bg-green-500/10 border-green-500/30'}`}>
                            <div className="text-xs uppercase font-medium opacity-70 mb-1">Anomalies Detected</div>
                            <div className={`text-2xl font-bold ${report.anomalies_detected > 0 ? 'text-red-400' : 'text-green-400'}`}>
                                {report.anomalies_detected}
                            </div>
                        </div>
                        <div className="p-4 rounded-lg bg-slate-700/30 border border-slate-600/30">
                            <div className="text-xs uppercase font-medium opacity-70 mb-1 text-slate-400">Actions Taken</div>
                            <div className="text-2xl font-bold text-white">{report.actions_taken?.length || 0}</div>
                        </div>
                    </div>

                    {report.actions_taken?.length > 0 && (
                        <div className="bg-slate-900/50 rounded-lg p-4 font-mono text-xs text-slate-300 border border-slate-700">
                            <div className="mb-2 text-primary-400 font-bold uppercase tracking-wider">Healing Log:</div>
                            <ul className="space-y-1">
                                {report.actions_taken.map((action, i) => (
                                    <li key={i} className="flex gap-2">
                                        <span className="text-green-500">➜</span> {action}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {report.anomalies_detected === 0 && (
                        <div className="flex items-center gap-2 text-green-400 bg-green-500/10 p-3 rounded-lg border border-green-500/20 justify-center">
                            <CheckCircle /> System Healthy. No actions needed.
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default SentinelPanel;
