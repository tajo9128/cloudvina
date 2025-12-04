import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
import { Activity, Users, Database, Cpu, CheckCircle, Clock, AlertTriangle, RefreshCw } from 'lucide-react';

const StatCard = ({ title, value, icon: Icon, subtext, color = "primary" }) => {
    const colorClasses = {
        primary: 'from-primary-500/20 to-primary-600/10 border-primary-500/30 text-primary-400',
        green: 'from-green-500/20 to-green-600/10 border-green-500/30 text-green-400',
        purple: 'from-secondary-500/20 to-secondary-600/10 border-secondary-500/30 text-secondary-400',
        orange: 'from-orange-500/20 to-orange-600/10 border-orange-500/30 text-orange-400',
        red: 'from-red-500/20 to-red-600/10 border-red-500/30 text-red-400',
    };

    return (
        <div className={`bg-gradient-to-br ${colorClasses[color]} border rounded-xl p-6 backdrop-blur-sm`}>
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-slate-300 text-sm font-medium">{title}</h3>
                <div className={`p-2 rounded-lg bg-slate-800/50`}>
                    <Icon size={20} className={colorClasses[color].split(' ').pop()} />
                </div>
            </div>
            <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-white">{value}</span>
            </div>
            {subtext && <p className="text-slate-500 text-xs mt-2">{subtext}</p>}
        </div>
    );
};

const Dashboard = () => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);

    useEffect(() => {
        fetchStats();
        const interval = setInterval(fetchStats, 30000);
        return () => clearInterval(interval);
    }, []);

    const fetchStats = async (showRefresh = false) => {
        if (showRefresh) setRefreshing(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();

            const response = await fetch(`${import.meta.env.VITE_API_URL}/admin/dashboard-stats`, {
                headers: {
                    'Authorization': `Bearer ${session?.access_token}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                setStats(data);
            }
        } catch (error) {
            console.error('Error fetching dashboard stats:', error);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    if (loading) {
        return (
            <div className="p-8 flex items-center justify-center h-full">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
            </div>
        );
    }

    return (
        <div className="p-8">
            <header className="mb-8 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Dashboard Overview</h1>
                    <p className="text-slate-400">Real-time system monitoring and statistics</p>
                </div>
                <button
                    onClick={() => fetchStats(true)}
                    disabled={refreshing}
                    className="flex items-center gap-2 px-4 py-2 bg-primary-600/20 text-primary-400 border border-primary-500/30 rounded-lg hover:bg-primary-600/30 transition-colors disabled:opacity-50"
                >
                    <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
                    Refresh
                </button>
            </header>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard
                    title="Total Jobs (24h)"
                    value={stats?.jobs?.total || 0}
                    icon={Database}
                    subtext={`${stats?.jobs?.running || 0} running, ${stats?.jobs?.queued || 0} queued`}
                    color="primary"
                />
                <StatCard
                    title="Success Rate"
                    value={`${stats?.jobs?.total ? Math.round((stats.jobs.completed / stats.jobs.total) * 100) : 0}%`}
                    icon={CheckCircle}
                    subtext={`${stats?.jobs?.failed || 0} failed jobs`}
                    color="green"
                />
                <StatCard
                    title="Active Users"
                    value={stats?.users?.active_today || 0}
                    icon={Users}
                    subtext={`Total users: ${stats?.users?.total || 0}`}
                    color="purple"
                />
                <StatCard
                    title="System Load"
                    value={`${stats?.system?.cpu_percent?.toFixed(1) || 0}%`}
                    icon={Cpu}
                    subtext={`Memory: ${stats?.system?.memory_percent?.toFixed(1) || 0}%`}
                    color="orange"
                />
            </div>

            {/* Charts/Details Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-slate-800/30 border border-primary-500/20 rounded-xl p-6 backdrop-blur-sm">
                    <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                        <Activity className="text-primary-400" size={20} />
                        Job Status Distribution
                    </h2>
                    <div className="space-y-4">
                        {['completed', 'failed', 'running', 'queued'].map(status => {
                            const count = stats?.jobs?.[status] || 0;
                            const total = stats?.jobs?.total || 1;
                            const percentage = Math.round((count / total) * 100);

                            const colors = {
                                completed: 'bg-green-500',
                                failed: 'bg-red-500',
                                running: 'bg-primary-500',
                                queued: 'bg-yellow-500'
                            };

                            return (
                                <div key={status}>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-slate-300 capitalize">{status}</span>
                                        <span className="text-slate-400">{count} ({percentage}%)</span>
                                    </div>
                                    <div className="w-full bg-slate-700/50 rounded-full h-2">
                                        <div className={`${colors[status]} h-2 rounded-full transition-all duration-500`} style={{ width: `${percentage}%` }}></div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                <div className="bg-slate-800/30 border border-primary-500/20 rounded-xl p-6 backdrop-blur-sm">
                    <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                        <Cpu className="text-primary-400" size={20} />
                        System Health
                    </h2>
                    <div className="space-y-4">
                        <div className="flex justify-between items-center py-3 border-b border-slate-700/50">
                            <span className="text-slate-300">API Status</span>
                            <span className="px-3 py-1 bg-green-500/10 text-green-400 rounded-full text-xs font-bold flex items-center gap-1">
                                <CheckCircle size={12} /> OPERATIONAL
                            </span>
                        </div>
                        <div className="flex justify-between items-center py-3 border-b border-slate-700/50">
                            <span className="text-slate-300">Database Connection</span>
                            <span className="px-3 py-1 bg-green-500/10 text-green-400 rounded-full text-xs font-bold flex items-center gap-1">
                                <CheckCircle size={12} /> CONNECTED
                            </span>
                        </div>
                        <div className="flex justify-between items-center py-3 border-b border-slate-700/50">
                            <span className="text-slate-300">AWS Batch Queue</span>
                            <span className="px-3 py-1 bg-primary-500/10 text-primary-400 rounded-full text-xs font-bold flex items-center gap-1">
                                <Clock size={12} /> ACTIVE
                            </span>
                        </div>
                        <div className="flex justify-between items-center py-3">
                            <span className="text-slate-300">Disk Usage</span>
                            <span className="text-slate-400 font-mono">{stats?.system?.disk_percent?.toFixed(1) || 0}%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
