import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
import { Database, Activity, GitCommit, FileText, Settings, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

const PhaseCard = ({ id, phase, config, onToggle, onUpdateLimit }) => {
    const [limit, setLimit] = useState(config.limit_per_user || 0);
    const [isEditing, setIsEditing] = useState(false);

    const handleSaveLimit = () => {
        onUpdateLimit(id, limit);
        setIsEditing(false);
    };

    const icons = {
        docking: Database,
        md_simulation: Activity,
        trajectory_analysis: GitCommit,
        binding_energy: Activity,
        lead_ranking: Settings,
        admet_prediction: FileText,
        target_prediction: CheckCircle,
        benchmarking: BarChart2,
        reporting: FileText
    };
    const Icon = icons[id] || Settings;

    return (
        <div className={`border rounded-xl p-5 ${config.enabled ? 'bg-slate-800/40 border-slate-700' : 'bg-slate-900/40 border-slate-800 opacity-75'}`}>
            <div className="flex justify-between items-start mb-4">
                <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${config.enabled ? 'bg-indigo-500/20 text-indigo-400' : 'bg-slate-700/50 text-slate-500'}`}>
                        <Icon size={20} />
                    </div>
                    <div>
                        <h3 className="text-white font-medium capitalize">{id.replace('_', ' ')}</h3>
                        <p className="text-xs text-slate-400">{config.maintenance_mode ? '⚠️ Maintenance' : (config.enabled ? 'Active' : 'Disabled')}</p>
                    </div>
                </div>
                <button
                    onClick={() => onToggle(id, 'enabled')}
                    className={`w-10 h-6 rounded-full transition-colors relative flex items-center ${config.enabled ? 'bg-green-500' : 'bg-slate-600'}`}
                >
                    <span className={`w-4 h-4 bg-white rounded-full shadow-sm transform transition-transform ml-1 ${config.enabled ? 'translate-x-4' : ''}`} />
                </button>
            </div>

            <div className="space-y-4">
                {/* Maintenance Toggle */}
                <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-400">Maintenance Mode</span>
                    <button
                        onClick={() => onToggle(id, 'maintenance_mode')}
                        className={`text-xs px-2 py-1 rounded border transition-colors ${config.maintenance_mode
                            ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                            : 'bg-slate-700/50 text-slate-400 border-slate-600'
                            }`}
                    >
                        {config.maintenance_mode ? 'On' : 'Off'}
                    </button>
                </div>

                {/* Quota Limit */}
                {config.limit_per_user !== undefined && (
                    <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-400">Default Limit / User</span>
                        {isEditing ? (
                            <div className="flex items-center gap-2">
                                <input
                                    type="number"
                                    value={limit}
                                    onChange={(e) => setLimit(parseInt(e.target.value))}
                                    className="w-16 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-white text-xs"
                                />
                                <button onClick={handleSaveLimit} className="text-green-400 hover:text-green-300"><CheckCircle size={14} /></button>
                                <button onClick={() => setIsEditing(false)} className="text-red-400 hover:text-red-300"><XCircle size={14} /></button>
                            </div>
                        ) : (
                            <button onClick={() => setIsEditing(true)} className="text-white font-mono bg-slate-700/50 px-2 py-1 rounded hover:bg-slate-700 transition-colors">
                                {config.limit_per_user}
                            </button>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

const PhasesControl = () => {
    const [config, setConfig] = useState({});
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            const res = await fetch(`${apiUrl}/admin/settings`, {
                headers: { 'Authorization': `Bearer ${session?.access_token}` }
            });
            const data = await res.json();
            setConfig(data.phases || {});
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const updatePhase = async (phaseId, field, value) => {
        const newPhaseConfig = { ...config[phaseId], [field]: value };
        // Optimistic update
        setConfig(prev => ({ ...prev, [phaseId]: newPhaseConfig }));

        try {
            const { data: { session } } = await supabase.auth.getSession();
            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            await fetch(`${apiUrl}/admin/settings/phases`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session?.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ [phaseId]: newPhaseConfig })
            });
        } catch (err) {
            console.error("Failed to update settings", err);
            fetchSettings(); // Revert
        }
    };

    const handleToggle = (id, field) => {
        updatePhase(id, field, !config[id][field]);
    };

    return (
        <div className="p-6 md:p-8 max-w-[1920px] mx-auto space-y-6">
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-2">Phase Control</h1>
                <p className="text-slate-400">Manage availability and quotas for all 9 pipeline phases.</p>
            </header>

            {loading ? (
                <div className="text-slate-400">Loading configurations...</div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {Object.entries(config).map(([id, conf]) => (
                        <PhaseCard
                            key={id}
                            id={id}
                            phase={id}
                            config={conf}
                            onToggle={handleToggle}
                            onUpdateLimit={(id, val) => updatePhase(id, 'limit_per_user', val)}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};

export default PhasesControl;
