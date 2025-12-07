import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
import { Link } from 'react-router-dom';

// Simple Icons (Replacing Lucide for now to match style or avoid missing imports/unused vars if any)
const Activity = () => <span>📈</span>;
const Users = () => <span>👥</span>;
const Database = () => <span>💾</span>;
const Cpu = () => <span>🖥️</span>;
const CheckCircle = () => <span>✅</span>;
const Clock = () => <span>🕒</span>;
const AlertTriangle = () => <span>⚠️</span>;
const RefreshCw = () => <span>🔄</span>;
const Zap = () => <span>⚡</span>;
const Shield = () => <span>🛡️</span>;
const Play = () => <span>▶️</span>;
const XCircle = () => <span>❌</span>;
const Plus = () => <span>➕</span>;
const Search = () => <span>🔍</span>;

const StatCard = ({ title, value, icon: Icon, subtext, color = "primary", trend }) => {
    const colorClasses = {
        primary: 'from-primary-500/20 to-primary-600/10 border-primary-500/30 text-primary-400',
        green: 'from-green-500/20 to-green-600/10 border-green-500/30 text-green-400',
        purple: 'from-secondary-500/20 to-secondary-600/10 border-secondary-500/30 text-secondary-400',
        orange: 'from-orange-500/20 to-orange-600/10 border-orange-500/30 text-orange-400',
        red: 'from-red-500/20 to-red-600/10 border-red-500/30 text-red-400',
    };

    return (
        <div className={`bg-gradient-to-br ${colorClasses[color]} border rounded-xl p-6 backdrop-blur-sm relative overflow-hidden group hover:scale-[1.02] transition-transform duration-300`}>
            <div className="absolute -right-6 -top-6 opacity-10 group-hover:opacity-20 transition-opacity">
                <Icon size={100} />
            </div>
            <div className="flex items-center justify-between mb-4 relative z-10">
                <h3 className="text-slate-300 text-sm font-medium uppercase tracking-wider">{title}</h3>
                <div className={`p-2 rounded-lg bg-slate-800/50 shadow-lg`}>
                    <Icon size={20} className={colorClasses[color].split(' ').pop()} />
                </div>
            </div>
            <div className="flex items-baseline gap-2 relative z-10">
                <span className="text-3xl font-bold text-white tracking-tight">{value}</span>
                {trend && <span className="text-xs font-medium text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded-full">{trend}</span>}
            </div>
            {subtext && <p className="text-slate-400 text-xs mt-2 relative z-10 font-medium">{subtext}</p>}
        </div>
    );
};

const Dashboard = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 15000); // Poll every 15s
        return () => clearInterval(interval);
    }, []);

    const fetchData = async (showRefresh = false) => {
        if (showRefresh) setRefreshing(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const response = await fetch(`${import.meta.env.VITE_API_URL}/admin/dashboard-stats`, {
                headers: { 'Authorization': `Bearer ${session?.access_token}` }
            });

            if (response.ok) {
                const result = await response.json();
                setData(result);
            }
        } catch (error) {
            console.error('Error fetching dashboard:', error);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    const handleCancelJob = async (jobId) => {
        if (!confirm('Cancel this job?')) return;
        try {
            const { data: { session } } = await supabase.auth.getSession();
            await fetch(`${import.meta.env.VITE_API_URL}/admin/jobs/${jobId}/cancel`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${session?.access_token}` }
            });
            fetchData();
        } catch (e) {
            console.error(e);
        }
    };

    if (loading) return <div className="p-20 text-center text-slate-400">Loading Command Center...</div>;

    const stats = data?.stats || {};
    const recentJobs = data?.recent_jobs || [];
    const recentUsers = data?.recent_users || [];
    const activityLog = data?.activity_log || [];

    const getStatusBadge = (status) => {
        const colors = {
            running: 'text-primary-400 bg-primary-500/10 border-primary-500/30',
            completed: 'text-green-400 bg-green-500/10 border-green-500/30',
            failed: 'text-red-400 bg-red-500/10 border-red-500/30',
            queued: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30'
        };
        const icons = { running: RefreshCw, completed: CheckCircle, failed: XCircle, queued: Clock };
        const Icon = icons[status] || Activity;

        return (
            <span className={`px-2 py-1 rounded-full text-xs font-medium border flex items-center gap-1 w-fit ${colors[status] || 'text-slate-400 bg-slate-500/10'}`}>
                <Icon size={12} className={status === 'running' ? 'animate-spin' : ''} />
                <span className="capitalize">{status}</span>
            </span>
        );
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-3xl font-bold text-white flex items-center gap-2">
                        <Zap className="text-secondary-400 fill-secondary-400" /> Command Center
                    </h1>
                    <p className="text-slate-400 text-sm">Real-time system overview & control</p>
                </div>
                <div className="flex gap-3">
                    <button onClick={() => fetchData(true)} className="p-2 bg-slate-800 text-slate-400 hover:text-white rounded-lg border border-slate-700 hover:border-slate-600 transition-all">
                        <RefreshCw size={20} className={refreshing ? 'animate-spin' : ''} />
                    </button>
                </div>
            </div>

            {/* Stats Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard title="Active Jobs" value={(stats.jobs?.running || 0) + (stats.jobs?.queued || 0)} icon={Database} subtext={`${stats.jobs?.total || 0} total today`} color="primary" trend="+12%" />
                <StatCard title="Success Rate" value={`${Math.round((stats.jobs?.completed / (stats.jobs?.total || 1)) * 100)}%`} icon={CheckCircle} subtext={`${stats.jobs?.failed || 0} failed jobs`} color="green" />
                <StatCard title="Total Users" value={stats.users?.total || 0} icon={Users} subtext="Lifetime registrations" color="purple" trend="+5 new" />
                <StatCard title="Server Load" value={`${stats.system?.cpu_percent || 0}%`} icon={Cpu} subtext={`Mem: ${stats.system?.memory_percent || 0}%`} color="orange" />
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Col: Live Jobs */}
                <div className="lg:col-span-2 bg-slate-800/40 border border-primary-500/20 rounded-xl backdrop-blur-sm overflow-hidden flex flex-col">
                    <div className="p-5 border-b border-primary-500/10 flex justify-between items-center bg-slate-800/40">
                        <h2 className="text-lg font-bold text-white flex items-center gap-2">
                            <Activity className="text-primary-400" size={18} /> Live Job Queue
                        </h2>
                        <Link to="/admin/jobs" className="text-xs text-primary-400 hover:text-primary-300 flex items-center gap-1">View All <Play size={10} /></Link>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead className="bg-slate-900/40 text-xs uppercase text-slate-500 font-medium">
                                <tr>
                                    <th className="p-4">ID</th>
                                    <th className="p-4">User</th>
                                    <th className="p-4">Target</th>
                                    <th className="p-4">Status</th>
                                    <th className="p-4 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-primary-500/5 text-sm">
                                {recentJobs.length === 0 ? (
                                    <tr><td colSpan="5" className="p-8 text-center text-slate-500">No active jobs</td></tr>
                                ) : (
                                    recentJobs.map(job => (
                                        <tr key={job.id} className="hover:bg-primary-500/5 transition-colors">
                                            <td className="p-4 font-mono text-xs text-slate-400">{job.id.substring(0, 8)}</td>
                                            <td className="p-4 text-white font-medium">{job.profiles?.email?.split('@')[0]}</td>
                                            <td className="p-4 text-slate-300">{job.receptor_filename || 'Unknown'}</td>
                                            <td className="p-4">{getStatusBadge(job.status)}</td>
                                            <td className="p-4 text-right">
                                                {(job.status === 'running' || job.status === 'queued') && (
                                                    <button onClick={() => handleCancelJob(job.id)} className="text-red-400 hover:text-red-300 p-1 hover:bg-red-500/10 rounded transition-colors" title="Cancel Job">
                                                        <XCircle size={16} />
                                                    </button>
                                                )}
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Right Col: System & Activity */}
                <div className="space-y-6">
                    {/* Activity Feed */}
                    <div className="bg-slate-800/40 border border-primary-500/20 rounded-xl backdrop-blur-sm p-5 h-[400px] flex flex-col">
                        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                            <Shield className="text-orange-400" size={18} /> Admin Activity
                        </h2>
                        <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar">
                            {activityLog.length === 0 ? (
                                <p className="text-slate-500 text-sm text-center py-10">No recent activity</p>
                            ) : (
                                activityLog.map((log, i) => (
                                    <div key={i} className="flex gap-3 text-sm group">
                                        <div className="mt-1">
                                            <div className="w-2 h-2 rounded-full bg-slate-600 group-hover:bg-primary-400 transition-colors"></div>
                                        </div>
                                        <div>
                                            <p className="text-slate-300">
                                                <span className="text-white font-medium capitalize">{log.action_type.replace('_', ' ')}</span>
                                                <span className="text-slate-500"> on {log.target_type}</span>
                                            </p>
                                            <p className="text-slate-500 text-xs">
                                                {new Date(log.created_at).toLocaleTimeString()} • {log.ip_address || 'Unknown IP'}
                                            </p>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Quick Users */}
                    <div className="bg-slate-800/40 border border-primary-500/20 rounded-xl backdrop-blur-sm p-5">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-lg font-bold text-white flex items-center gap-2">
                                <Users className="text-purple-400" size={18} /> New Users
                            </h2>
                            <Link to="/admin/users" className="text-xs text-purple-400 hover:text-purple-300">View All</Link>
                        </div>
                        <div className="space-y-3">
                            {recentUsers.map(user => (
                                <div key={user.id} className="flex items-center justify-between p-2 hover:bg-white/5 rounded-lg transition-colors">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500/30 to-blue-500/30 flex items-center justify-center text-xs font-bold text-white">
                                            {user.email?.charAt(0).toUpperCase()}
                                        </div>
                                        <div>
                                            <p className="text-sm text-white font-medium">{user.email?.split('@')[0]}</p>
                                            <p className="text-xs text-slate-500">{new Date(user.created_at).toLocaleDateString()}</p>
                                        </div>
                                    </div>
                                    <span className={`text-[10px] px-2 py-0.5 rounded border ${user.is_admin ? 'bg-secondary-500/10 text-secondary-400 border-secondary-500/30' : 'bg-slate-700/50 text-slate-400 border-slate-600'}`}>
                                        {user.is_admin ? 'ADMIN' : 'USER'}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
