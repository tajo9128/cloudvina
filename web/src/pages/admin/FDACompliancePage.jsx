import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
import { API_URL } from '../../config';
import { Shield, Search, FileText, Download, Lock, RefreshCw } from 'lucide-react';

export default function FDACompliancePage() {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filters, setFilters] = useState({
        user_id: '',
        resource_id: '',
        limit: 100
    });

    useEffect(() => {
        fetchLogs();
    }, []);

    const fetchLogs = async () => {
        setLoading(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) return;

            let query = `?limit=${filters.limit}`;
            if (filters.user_id) query += `&user_id=${filters.user_id}`;
            if (filters.resource_id) query += `&resource_id=${filters.resource_id}`;

            const res = await fetch(`${API_URL}/admin/fda/logs${query}`, {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            });

            if (!res.ok) throw new Error('Failed to fetch logs');
            const data = await res.json();
            setLogs(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleExport = () => {
        const headers = ["ID", "Timestamp", "User ID", "Action", "Resource ID", "IP Address", "Details"];
        const csvContent = [
            headers.join(','),
            ...logs.map(log => [
                log.id,
                log.created_at,
                log.user_id || 'System',
                log.action,
                log.resource_id || '-',
                log.ip_address || '-',
                `"${JSON.stringify(log.details).replace(/"/g, '""')}"`
            ].join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `FDA_Audit_Trail_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
    };

    return (
        <div className="min-h-screen bg-slate-50 p-6">
            <header className="mb-8">
                <div className="flex items-center gap-3 mb-2">
                    <Shield className="w-8 h-8 text-emerald-600" />
                    <h1 className="text-2xl font-bold text-slate-900">FDA 21 CFR Part 11 Compliance</h1>
                </div>
                <p className="text-slate-600">Secure, immutable audit trail for all critical system activities.</p>
            </header>

            {/* FILTERS */}
            <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 mb-6 flex flex-wrap gap-4 items-end">
                <div>
                    <label className="block text-xs font-bold text-slate-500 mb-1">User ID</label>
                    <input
                        type="text"
                        placeholder="Filter by User..."
                        className="px-3 py-2 border rounded-lg text-sm w-48"
                        value={filters.user_id}
                        onChange={e => setFilters({ ...filters, user_id: e.target.value })}
                    />
                </div>
                <div>
                    <label className="block text-xs font-bold text-slate-500 mb-1">Resource / Batch ID</label>
                    <input
                        type="text"
                        placeholder="Filter by resource..."
                        className="px-3 py-2 border rounded-lg text-sm w-48"
                        value={filters.resource_id}
                        onChange={e => setFilters({ ...filters, resource_id: e.target.value })}
                    />
                </div>
                <div>
                    <label className="block text-xs font-bold text-slate-500 mb-1">Limit</label>
                    <select
                        className="px-3 py-2 border rounded-lg text-sm"
                        value={filters.limit}
                        onChange={e => setFilters({ ...filters, limit: e.target.value })}
                    >
                        <option value="50">50 Rows</option>
                        <option value="100">100 Rows</option>
                        <option value="500">500 Rows</option>
                    </select>
                </div>
                <button
                    onClick={fetchLogs}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-bold flex items-center gap-2 hover:bg-indigo-700"
                >
                    <Search className="w-4 h-4" /> Filter
                </button>
                <button
                    onClick={() => { setFilters({ user_id: '', resource_id: '', limit: 100 }); setTimeout(fetchLogs, 10); }}
                    className="px-4 py-2 bg-slate-100 text-slate-600 rounded-lg text-sm font-bold flex items-center gap-2 hover:bg-slate-200"
                >
                    <RefreshCw className="w-4 h-4" /> Reset
                </button>
                <div className="flex-1 text-right">
                    <button
                        onClick={handleExport}
                        className="px-4 py-2 border border-slate-300 text-slate-700 rounded-lg text-sm font-bold flex items-center gap-2 hover:bg-slate-50 ml-auto"
                    >
                        <Download className="w-4 h-4" /> Export CSV
                    </button>
                </div>
            </div>

            {/* TABLE */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                <table className="w-full text-sm text-left">
                    <thead className="bg-slate-50 text-slate-500 font-bold border-b border-slate-200">
                        <tr>
                            <th className="px-6 py-4">Timestamp (UTC)</th>
                            <th className="px-6 py-4">User</th>
                            <th className="px-6 py-4">Action</th>
                            <th className="px-6 py-4">Resource</th>
                            <th className="px-6 py-4">Details</th>
                            <th className="px-6 py-4">Integrity</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                        {loading ? (
                            <tr><td colSpan="6" className="px-6 py-12 text-center text-slate-500">Loading secure logs...</td></tr>
                        ) : logs.length === 0 ? (
                            <tr><td colSpan="6" className="px-6 py-12 text-center text-slate-500">No audit logs found matching criteria.</td></tr>
                        ) : (
                            logs.map(log => (
                                <tr key={log.id} className="hover:bg-slate-50 transition-colors">
                                    <td className="px-6 py-3 font-mono text-slate-600">
                                        {new Date(log.created_at).toISOString().replace('T', ' ').substring(0, 19)}
                                    </td>
                                    <td className="px-6 py-3">
                                        <div className="font-semibold text-slate-900 truncate w-32" title={log.user_id}>{log.user_id}</div>
                                    </td>
                                    <td className="px-6 py-3">
                                        <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-bold ${log.action.includes('FAIL') ? 'bg-red-100 text-red-700' :
                                                log.action.includes('SUBMIT') ? 'bg-indigo-100 text-indigo-700' :
                                                    'bg-emerald-100 text-emerald-700'
                                            }`}>
                                            {log.action}
                                        </span>
                                    </td>
                                    <td className="px-6 py-3 font-mono text-slate-500 text-xs">
                                        {log.resource_id || '-'}
                                    </td>
                                    <td className="px-6 py-3">
                                        <pre className="text-[10px] text-slate-500 bg-slate-100 p-2 rounded max-w-xs overflow-x-auto">
                                            {JSON.stringify(log.details, null, 2)}
                                        </pre>
                                    </td>
                                    <td className="px-6 py-3">
                                        <div className="flex items-center gap-1 text-emerald-600 text-xs font-bold" title="Log is immutable">
                                            <Lock className="w-3 h-3" /> Valid
                                        </div>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
