import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
import { API_URL } from '../../config';
import { Line, Scatter } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
} from 'chart.js';
import { Target, Upload, FileText, ChevronRight, Activity, TrendingUp } from 'lucide-react';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

export default function BenchmarkingPage() {
    const [batches, setBatches] = useState([]);
    const [selectedBatch, setSelectedBatch] = useState('');
    const [file, setFile] = useState(null);
    const [analysisName, setAnalysisName] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [history, setHistory] = useState([]);

    useEffect(() => {
        fetchBatches();
        fetchHistory();
    }, []);

    const fetchBatches = async () => {
        const { data: { session } } = await supabase.auth.getSession();
        if (!session) return;

        // Fetch last 20 batches
        const { data } = await supabase
            .from('jobs')
            .select('batch_id, created_at')
            .not('batch_id', 'is', null)
            .order('created_at', { ascending: false })
            .limit(50);

        // Unique batches
        const unique = [...new Map(data.map(item => [item.batch_id, item])).values()];
        setBatches(unique);
    };

    const fetchHistory = async () => {
        const { data: { session } } = await supabase.auth.getSession();
        if (!session) return;

        try {
            const res = await fetch(`${API_URL}/tools/benchmark/history`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            });
            const data = await res.json();
            setHistory(data);
        } catch (e) {
            console.error(e);
        }
    };

    const handleAnalyze = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const formData = new FormData();
            formData.append('batch_id', selectedBatch);
            formData.append('name', analysisName);
            formData.append('file', file);

            const res = await fetch(`${API_URL}/tools/benchmark/analyze`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${session.access_token}` },
                body: formData
            });

            if (!res.ok) throw new Error(await res.text());

            const data = await res.json();
            setResult(data);
            fetchHistory(); // refresh list
        } catch (err) {
            alert("Analysis Failed: " + err.message);
        } finally {
            setLoading(false);
        }
    };

    const loadAnalysis = async (id) => {
        setLoading(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const res = await fetch(`${API_URL}/tools/benchmark/${id}`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            });
            const data = await res.json();
            setResult(data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    // Chart Options
    const chartOptions = {
        responsive: true,
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: 'Experimental vs Predicted Affinity' },
            tooltip: {
                callbacks: {
                    label: (ctx) => {
                        const point = ctx.raw;
                        return `${point.name}: Exp ${point.x}, Pred ${point.y}`;
                    }
                }
            }
        },
        scales: {
            x: {
                title: { display: true, text: 'Experimental Value (Reference)' },
                type: 'linear',
                position: 'bottom'
            },
            y: {
                title: { display: true, text: 'Predicted Binding Affinity (kcal/mol)' },
                type: 'linear'
            }
        }
    };

    const chartData = {
        datasets: [
            {
                label: 'Compounds',
                data: result?.plot_data || [],
                backgroundColor: 'rgba(79, 70, 229, 0.7)',
                borderColor: 'rgba(79, 70, 229, 1)',
                pointRadius: 6,
                pointHoverRadius: 8
            },
            // Ideal Line (y=x) - Simplified
            {
                label: 'Ideal Fit',
                data: result ? [
                    { x: Math.min(...result.plot_data.map(p => p.x)), y: Math.min(...result.plot_data.map(p => p.x)) },
                    { x: Math.max(...result.plot_data.map(p => p.x)), y: Math.max(...result.plot_data.map(p => p.x)) }
                ] : [],
                borderColor: 'rgba(200, 200, 200, 0.5)',
                borderDash: [5, 5],
                pointRadius: 0,
                showLine: true,
                type: 'line',
                fill: false
            }
        ],
    };

    return (
        <div className="min-h-screen bg-slate-50 p-6">
            <header className="mb-8">
                <div className="flex items-center gap-3 mb-2">
                    <Activity className="w-8 h-8 text-indigo-600" />
                    <h1 className="text-2xl font-bold text-slate-900">Accuracy Benchmarking</h1>
                </div>
                <p className="text-slate-600">Validate engine performance against experimental datasets (e.g., PDBbind, ChEMBL).</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* LEFT: New Analysis Form */}
                <div className="lg:col-span-1 space-y-6">
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                        <h3 className="font-bold text-lg mb-4 flex items-center gap-2"><Target className="w-5 h-5 text-emerald-600" /> New Analysis</h3>
                        <form onSubmit={handleAnalyze} className="space-y-4">
                            <div>
                                <label className="block text-xs font-bold text-slate-500 mb-1">Batch Job (Predictions)</label>
                                <select
                                    className="w-full p-2 border rounded-lg text-sm bg-slate-50"
                                    value={selectedBatch}
                                    onChange={e => setSelectedBatch(e.target.value)}
                                    required
                                >
                                    <option value="">Select a batch...</option>
                                    {batches.map(b => (
                                        <option key={b.batch_id} value={b.batch_id}>
                                            {b.batch_id.substring(0, 8)} - (Ref: {new Date(b.created_at).toLocaleDateString()})
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className="block text-xs font-bold text-slate-500 mb-1">Report Name</label>
                                <input
                                    type="text"
                                    className="w-full p-2 border rounded-lg text-sm"
                                    placeholder="e.g. HIV Protease Validation"
                                    value={analysisName}
                                    onChange={e => setAnalysisName(e.target.value)}
                                    required
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-bold text-slate-500 mb-1">Reference Data (.csv)</label>
                                <div className="border-2 border-dashed border-slate-200 rounded-lg p-4 text-center hover:bg-slate-50 transition cursor-pointer relative">
                                    <input
                                        type="file"
                                        accept=".csv"
                                        onChange={e => setFile(e.target.files[0])}
                                        className="absolute inset-0 opacity-0 cursor-pointer"
                                        required
                                    />
                                    <div className="pointer-events-none">
                                        <Upload className="w-6 h-6 text-slate-400 mx-auto mb-2" />
                                        <div className="text-xs text-slate-500">
                                            {file ? file.name : "Upload CSV (name, value)"}
                                        </div>
                                    </div>
                                </div>
                                <p className="text-[10px] text-slate-400 mt-1">Columns required: <code>name</code> (ligand ID), <code>value</code> (Exp. Affinity)</p>
                            </div>

                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full py-2 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 disabled:opacity-50 flex justify-center items-center gap-2"
                            >
                                {loading ? "Analyzing..." : "Run Analysis"} <TrendingUp className="w-4 h-4" />
                            </button>
                        </form>
                    </div>

                    {/* History List */}
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                        <h3 className="font-bold text-sm text-slate-500 uppercase mb-4">Past Validations</h3>
                        <div className="space-y-2 max-h-64 overflow-y-auto">
                            {history.map(item => (
                                <div
                                    key={item.id}
                                    onClick={() => loadAnalysis(item.id)}
                                    className="p-3 rounded-lg border border-slate-100 hover:bg-slate-50 cursor-pointer transition flex justify-between items-center group"
                                >
                                    <div>
                                        <div className="text-sm font-bold text-slate-800">{item.name}</div>
                                        <div className="text-xs text-slate-500">R²: {item.metrics?.r2 || 'N/A'} • n={item.metrics?.n}</div>
                                    </div>
                                    <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-indigo-500" />
                                </div>
                            ))}
                            {history.length === 0 && <div className="text-xs text-slate-400 text-center py-4">No history yet.</div>}
                        </div>
                    </div>
                </div>

                {/* RIGHT: Visualization */}
                <div className="lg:col-span-2">
                    {result ? (
                        <div className="space-y-6">
                            {/* Score Cards */}
                            <div className="grid grid-cols-3 gap-4">
                                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-100">
                                    <div className="text-xs font-bold text-slate-400 uppercase">R-Squared (R²)</div>
                                    <div className={`text-2xl font-bold ${result.metrics.r2 > 0.6 ? 'text-emerald-600' : 'text-amber-500'}`}>
                                        {result.metrics.r2}
                                    </div>
                                </div>
                                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-100">
                                    <div className="text-xs font-bold text-slate-400 uppercase">RMSE</div>
                                    <div className="text-2xl font-bold text-slate-700">
                                        {result.metrics.rmse}
                                    </div>
                                </div>
                                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-100">
                                    <div className="text-xs font-bold text-slate-400 uppercase">Samples (n)</div>
                                    <div className="text-2xl font-bold text-slate-700">
                                        {result.metrics.n}
                                    </div>
                                </div>
                            </div>

                            {/* Chart */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                                <Scatter options={chartOptions} data={chartData} height={150} />
                            </div>
                        </div>
                    ) : (
                        <div className="h-full bg-white rounded-xl border border-dashed border-slate-300 flex items-center justify-center p-12 text-slate-400">
                            Select a batch and upload a CSV to visualize accuracy.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
