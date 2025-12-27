import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
import { Settings, Save, RefreshCw, Server, Shield, Bell } from 'lucide-react';

const AdminSettings = () => {
    const [config, setConfig] = useState(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        fetchConfig();
    }, []);

    const fetchConfig = async () => {
        setLoading(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const res = await fetch(`${import.meta.env.VITE_API_URL}/admin/system/config`, {
                headers: { 'Authorization': `Bearer ${session?.access_token}` }
            });
            if (res.ok) {
                const data = await res.json();
                setConfig(data);
            }
        } catch (error) {
            console.error('Error fetching config:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async (e) => {
        e.preventDefault();
        setSaving(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            await fetch(`${import.meta.env.VITE_API_URL}/admin/system/config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session?.access_token}`
                },
                body: JSON.stringify(config)
            });
            alert('System configuration saved successfully.');
        } catch (error) {
            console.error('Save failed:', error);
            alert('Failed to save configuration.');
        } finally {
            setSaving(false);
        }
    };

    if (loading) return <div className="p-8 text-slate-400">Loading system configuration...</div>;

    return (
        <div className="p-8 max-w-4xl mx-auto">
            <header className="mb-8 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
                        <Settings className="text-slate-400" /> System Settings
                    </h1>
                    <p className="text-slate-400">Global configuration for BioDockify Platform</p>
                </div>
                <button onClick={fetchConfig} className="p-2 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors text-slate-400">
                    <RefreshCw size={20} />
                </button>
            </header>

            <form onSubmit={handleSave} className="space-y-6">

                {/* General Section */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                    <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                        <Server size={20} className="text-blue-400" /> Infrastructure
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label className="block text-sm font-medium text-slate-400 mb-1">Max Concurrent Jobs (Global)</label>
                            <input
                                type="number"
                                value={config?.max_concurrent_jobs || 100}
                                onChange={e => setConfig({ ...config, max_concurrent_jobs: parseInt(e.target.value) })}
                                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-blue-500 focus:outline-none"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-400 mb-1">Job Timeout (Minutes)</label>
                            <input
                                type="number"
                                value={config?.job_timeout_minutes || 60}
                                onChange={e => setConfig({ ...config, job_timeout_minutes: parseInt(e.target.value) })}
                                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-blue-500 focus:outline-none"
                            />
                        </div>
                    </div>
                </div>

                {/* Security Section */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                    <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                        <Shield size={20} className="text-green-400" /> Security & Access
                    </h2>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-800">
                            <div>
                                <h3 className="text-white font-medium">Maintenance Mode</h3>
                                <p className="text-sm text-slate-500">Disable all new job submissions globally.</p>
                            </div>
                            <label className="relative inline-flex items-center cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={config?.maintenance_mode || false}
                                    onChange={e => setConfig({ ...config, maintenance_mode: e.target.checked })}
                                    className="sr-only peer"
                                />
                                <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                            </label>
                        </div>

                        <div className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-800">
                            <div>
                                <h3 className="text-white font-medium">Bypass Payment for Admins</h3>
                                <p className="text-sm text-slate-500">Allow admins to submit jobs without credit deduction.</p>
                            </div>
                            <label className="relative inline-flex items-center cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={config?.admin_bypass_payment || false}
                                    onChange={e => setConfig({ ...config, admin_bypass_payment: e.target.checked })}
                                    className="sr-only peer"
                                />
                                <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                            </label>
                        </div>
                    </div>
                </div>

                {/* Notifications Section */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                    <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                        <Bell size={20} className="text-yellow-400" /> System Announcements
                    </h2>
                    <div>
                        <label className="block text-sm font-medium text-slate-400 mb-1">Banner Message (Leave empty to disable)</label>
                        <input
                            type="text"
                            value={config?.system_banner_message || ''}
                            onChange={e => setConfig({ ...config, system_banner_message: e.target.value })}
                            placeholder="e.g. Scheduled maintenance at 02:00 UTC..."
                            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-yellow-500 focus:outline-none"
                        />
                    </div>
                </div>

                <div className="flex justify-end pt-4">
                    <button
                        type="submit"
                        disabled={saving}
                        className="flex items-center gap-2 px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white font-bold rounded-xl transition-all"
                    >
                        <Save size={20} />
                        {saving ? 'Saving...' : 'Save Configuration'}
                    </button>
                </div>

            </form>
        </div>
    );
};

export default AdminSettings;
