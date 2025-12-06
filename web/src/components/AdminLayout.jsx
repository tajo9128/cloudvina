import React from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
// import { LayoutDashboard, Users, Database, Settings, LogOut, Home } from 'lucide-react';
import { supabase } from '../supabaseClient';

const LayoutDashboard = () => <span>📊</span>;
const Users = () => <span>👥</span>;
const Database = () => <span>💾</span>;
const Settings = () => <span>⚙️</span>;
const LogOut = () => <span>🚪</span>;
const Home = () => <span>🏠</span>;

const AdminLayout = () => {
    const location = useLocation();

    const handleLogout = async () => {
        await supabase.auth.signOut();
        window.location.href = '/';
    };

    const navItems = [
        { path: '/admin', icon: LayoutDashboard, label: 'Dashboard' },
        { path: '/admin/jobs', icon: Database, label: 'Jobs' },
        { path: '/admin/users', icon: Users, label: 'Users' },
        { path: '/admin/settings', icon: Settings, label: 'Settings' },
    ];

    return (
        <div className="flex h-screen bg-gradient-to-br from-slate-900 via-[#0B1121] to-slate-900 text-white">
            {/* Sidebar */}
            <div className="w-64 bg-slate-900/50 backdrop-blur-sm border-r border-primary-500/20 flex flex-col">
                <div className="p-6 border-b border-primary-500/20">
                    <Link to="/" className="flex items-center gap-2 font-bold text-xl">
                        <span className="text-2xl">🧬</span>
                        <span className="bg-gradient-to-r from-primary-400 to-secondary-400 bg-clip-text text-transparent">Bio<span className="text-white">Dockify</span></span>
                    </Link>
                    <p className="text-xs text-primary-400/60 mt-1">Admin Panel</p>
                </div>

                <nav className="flex-1 p-4 space-y-2">
                    {navItems.map((item) => {
                        const Icon = item.icon;
                        const isActive = location.pathname === item.path;

                        return (
                            <Link
                                key={item.path}
                                to={item.path}
                                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${isActive
                                    ? 'bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-lg shadow-primary-500/20'
                                    : 'text-slate-400 hover:bg-primary-500/10 hover:text-primary-300'
                                    }`}
                            >
                                <Icon size={20} />
                                <span>{item.label}</span>
                            </Link>
                        );
                    })}
                </nav>

                <div className="p-4 border-t border-primary-500/20 space-y-2">
                    <Link
                        to="/"
                        className="flex items-center gap-3 px-4 py-3 w-full text-slate-400 hover:bg-primary-500/10 hover:text-primary-300 rounded-lg transition-colors"
                    >
                        <Home size={20} />
                        <span>Back to Site</span>
                    </Link>
                    <button
                        onClick={handleLogout}
                        className="flex items-center gap-3 px-4 py-3 w-full text-slate-400 hover:bg-red-500/10 hover:text-red-400 rounded-lg transition-colors"
                    >
                        <LogOut size={20} />
                        <span>Sign Out</span>
                    </button>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-auto">
                <Outlet />
            </div>
        </div>
    );
};

export default AdminLayout;
