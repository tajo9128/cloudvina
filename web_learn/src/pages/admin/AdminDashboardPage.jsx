import React from 'react';
import { Users, DollarSign, BookOpen, TrendingUp, MoreVertical } from 'lucide-react';

export default function AdminDashboardPage() {
    const stats = [
        { label: 'Total Users', value: '1,234', change: '+12%', icon: Users, color: 'blue' },
        { label: 'Total Revenue', value: '$12,345', change: '+8%', icon: DollarSign, color: 'green' },
        { label: 'Active Courses', value: '24', change: '+2', icon: BookOpen, color: 'purple' },
        { label: 'Completion Rate', value: '68%', change: '+5%', icon: TrendingUp, color: 'amber' },
    ];

    const recentUsers = [
        { name: 'John Doe', email: 'john@example.com', role: 'Student', date: '2 mins ago' },
        { name: 'Sarah Smith', email: 'sarah@example.com', role: 'Instructor', date: '1 hour ago' },
        { name: 'Mike Johnson', email: 'mike@example.com', role: 'Student', date: '3 hours ago' },
        { name: 'Emma Wilson', email: 'emma@example.com', role: 'Student', date: '5 hours ago' },
    ];

    return (
        <div className="space-y-6">
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {stats.map((stat, index) => (
                    <div key={index} className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                        <div className="flex justify-between items-start mb-4">
                            <div className={`p-3 rounded-lg bg-${stat.color}-50 text-${stat.color}-600`}>
                                <stat.icon className="w-6 h-6" />
                            </div>
                            <span className="px-2 py-1 bg-green-50 text-green-700 text-xs font-bold rounded-full">
                                {stat.change}
                            </span>
                        </div>
                        <h3 className="text-2xl font-bold text-slate-900 mb-1">{stat.value}</h3>
                        <p className="text-slate-500 text-sm">{stat.label}</p>
                    </div>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Recent Activity */}
                <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                    <div className="p-6 border-b border-slate-100 flex justify-between items-center">
                        <h3 className="font-bold text-slate-900">Recent Registrations</h3>
                        <button className="text-primary-600 text-sm font-medium hover:underline">View All</button>
                    </div>
                    <table className="w-full text-left text-sm">
                        <thead className="bg-slate-50 text-slate-500">
                            <tr>
                                <th className="px-6 py-4 font-medium">User</th>
                                <th className="px-6 py-4 font-medium">Role</th>
                                <th className="px-6 py-4 font-medium">Date</th>
                                <th className="px-6 py-4 font-medium">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {recentUsers.map((user, i) => (
                                <tr key={i} className="hover:bg-slate-50 transition-colors">
                                    <td className="px-6 py-4">
                                        <div>
                                            <div className="font-bold text-slate-900">{user.name}</div>
                                            <div className="text-slate-500 text-xs">{user.email}</div>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2 py-1 rounded text-xs font-bold uppercase tracking-wider ${user.role === 'Instructor' ? 'bg-purple-100 text-purple-700' : 'bg-slate-100 text-slate-600'}`}>
                                            {user.role}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-slate-500">{user.date}</td>
                                    <td className="px-6 py-4">
                                        <button className="text-slate-400 hover:text-slate-600">
                                            <MoreVertical className="w-4 h-4" />
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Quick Actions */}
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
                    <h3 className="font-bold text-slate-900 mb-6">Quick Actions</h3>
                    <div className="space-y-3">
                        <button className="w-full py-3 px-4 bg-primary-600 hover:bg-primary-700 text-white font-bold rounded-lg transition-colors shadow-sm">
                            Create New Course
                        </button>
                        <button className="w-full py-3 px-4 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 font-bold rounded-lg transition-colors">
                            Manage Users
                        </button>
                        <button className="w-full py-3 px-4 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 font-bold rounded-lg transition-colors">
                            Review Reported Content
                        </button>
                        <button className="w-full py-3 px-4 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 font-bold rounded-lg transition-colors">
                            System Settings
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
