import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Menu, X, ChevronDown, User, Settings, LogOut, HelpCircle, Bell } from 'lucide-react';

export default function Header() {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [featuresOpen, setFeaturesOpen] = useState(false);
    const [toolsOpen, setToolsOpen] = useState(false);
    const [userMenuOpen, setUserMenuOpen] = useState(false);

    // Mock user state - replace with real auth
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const user = {
        name: 'John Doe',
        email: 'john@example.com',
        avatar: null
    };

    const featuresItems = [
        {
            title: 'Molecular Docking',
            description: 'Cloud-based AutoDock Vina for protein-ligand docking',
            href: 'https://www.biodockify.com/features/molecular-docking',
            icon: '🧬'
        },
        {
            title: '3D Viewer',
            description: 'Interactive visualization of docking results',
            href: 'https://www.biodockify.com/features/3d-viewer',
            icon: '👁️'
        },
        {
            title: 'MD Simulation',
            description: 'Molecular dynamics simulation for validation',
            href: 'https://www.biodockify.com/features/md-simulation',
            icon: '⚛️'
        },
        {
            title: 'Target Prediction',
            description: 'AI-powered protein target identification',
            href: 'https://www.biodockify.com/features/target-prediction',
            icon: '🎯'
        },
        {
            title: 'Ranking & Leads',
            description: 'Automated compound ranking and lead selection',
            href: 'https://www.biodockify.com/features/ranking-leads',
            icon: '📊'
        }
    ];

    const toolsItems = [
        {
            title: 'Protein Preparation',
            description: 'Automated PDB cleaning, protonation, and PDBQT conversion',
            href: 'https://www.biodockify.com/tools/protein-preparation',
            icon: '🧬'
        },
        {
            title: 'Ligand Preparation',
            description: 'SDF/MOL2 to PDBQT, energy minimization',
            href: 'https://www.biodockify.com/tools/ligand-preparation',
            icon: '💊'
        },
        {
            title: 'Virtual Screening',
            description: 'Screen millions of compounds against your target',
            href: 'https://www.biodockify.com/tools/virtual-screening',
            icon: '🔬'
        },
        {
            title: 'Binding Site Detection',
            description: 'Automated cavity detection and grid box setup',
            href: 'https://www.biodockify.com/tools/binding-site-detection',
            icon: '🎯'
        },
        {
            title: 'Docking Job Submission',
            description: 'Run AutoDock Vina jobs with configurable parameters',
            href: 'https://www.biodockify.com/tools/docking-submission',
            icon: '⚡'
        },
        {
            title: 'Results Analysis',
            description: '3D visualization, H-bond analysis, and data export',
            href: 'https://www.biodockify.com/tools/results-analysis',
            icon: '📊'
        }
    ];

    return (
        <header className="sticky top-0 z-50 bg-white border-b border-slate-200 shadow-sm">
            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <div className="flex items-center">
                        <Link to="/" className="flex items-center space-x-2">
                            <span className="text-2xl">🧬</span>
                            <span className="text-xl font-bold font-display">
                                <span className="text-slate-900">Bio</span>
                                <span className="text-primary-600">Dockify</span>
                            </span>
                            <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-slate-100 text-slate-700 rounded">
                                Learn
                            </span>
                        </Link>
                    </div>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center space-x-1">
                        <Link
                            to="/"
                            className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium"
                        >
                            Home
                        </Link>

                        {/* Features Dropdown */}
                        <div className="relative">
                            <button
                                onMouseEnter={() => setFeaturesOpen(true)}
                                onMouseLeave={() => setFeaturesOpen(false)}
                                className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium flex items-center gap-1"
                            >
                                Features
                                <ChevronDown className={`w-4 h-4 transition-transform ${featuresOpen ? 'rotate-180' : ''}`} />
                            </button>

                            {/* Features Dropdown Menu */}
                            {featuresOpen && (
                                <div
                                    onMouseEnter={() => setFeaturesOpen(true)}
                                    onMouseLeave={() => setFeaturesOpen(false)}
                                    className="absolute left-0 mt-2 w-80 bg-white rounded-xl shadow-xl border border-slate-200 py-2 z-50"
                                >
                                    {featuresItems.map((feature, index) => (
                                        <a
                                            key={index}
                                            href={feature.href}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="block px-4 py-3 hover:bg-slate-50 transition-colors group"
                                        >
                                            <div className="flex items-start gap-3">
                                                <span className="text-2xl">{feature.icon}</span>
                                                <div>
                                                    <div className="font-semibold text-slate-900 group-hover:text-primary-600 transition-colors">
                                                        {feature.title}
                                                    </div>
                                                    <div className="text-xs text-slate-500 mt-0.5">
                                                        {feature.description}
                                                    </div>
                                                </div>
                                            </div>
                                        </a>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Tools Dropdown */}
                        <div className="relative">
                            <button
                                onMouseEnter={() => setToolsOpen(true)}
                                onMouseLeave={() => setToolsOpen(false)}
                                className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium flex items-center gap-1"
                            >
                                Tools
                                <ChevronDown className={`w-4 h-4 transition-transform ${toolsOpen ? 'rotate-180' : ''}`} />
                            </button>

                            {/* Tools Dropdown Menu */}
                            {toolsOpen && (
                                <div
                                    onMouseEnter={() => setToolsOpen(true)}
                                    onMouseLeave={() => setToolsOpen(false)}
                                    className="absolute left-0 mt-2 w-80 bg-white rounded-xl shadow-xl border border-slate-200 py-2 z-50"
                                >
                                    {toolsItems.map((tool, index) => (
                                        <a
                                            key={index}
                                            href={tool.href}
                                            className="block px-4 py-3 hover:bg-slate-50 transition-colors group"
                                        >
                                            <div className="flex items-start gap-3">
                                                <span className="text-2xl">{tool.icon}</span>
                                                <div>
                                                    <div className="font-semibold text-slate-900 group-hover:text-primary-600 transition-colors">
                                                        {tool.title}
                                                    </div>
                                                    <div className="text-xs text-slate-500 mt-0.5">
                                                        {tool.description}
                                                    </div>
                                                </div>
                                            </div>
                                        </a>
                                    ))}
                                </div>
                            )}
                        </div>

                        <Link
                            to="/blog"
                            className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium"
                        >
                            Blog
                        </Link>

                        <Link
                            to="/courses"
                            className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium"
                        >
                            Courses
                        </Link>

                        <Link
                            to="/community"
                            className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium"
                        >
                            Community
                        </Link>

                        {isLoggedIn && (
                            <Link
                                to="/admin"
                                className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium"
                            >
                                Admin
                            </Link>
                        )}
                    </div>

                    {/* Right Side: Notifications + User Menu */}
                    <div className="hidden md:flex items-center space-x-3">
                        {isLoggedIn ? (
                            <>
                                {/* Notifications */}
                                <button className="relative p-2 text-slate-600 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors">
                                    <Bell className="w-5 h-5" />
                                    <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
                                </button>

                                {/* Support */}
                                <Link
                                    to="/support"
                                    className="p-2 text-slate-600 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors"
                                    title="Support"
                                >
                                    <HelpCircle className="w-5 h-5" />
                                </Link>

                                {/* User Menu */}
                                <div className="relative">
                                    <button
                                        onClick={() => setUserMenuOpen(!userMenuOpen)}
                                        className="flex items-center gap-2 p-2 hover:bg-slate-50 rounded-lg transition-colors"
                                    >
                                        {user.avatar ? (
                                            <img src={user.avatar} alt={user.name} className="w-8 h-8 rounded-full" />
                                        ) : (
                                            <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center text-white font-semibold">
                                                {user.name.charAt(0)}
                                            </div>
                                        )}
                                        <ChevronDown className={`w-4 h-4 text-slate-600 transition-transform ${userMenuOpen ? 'rotate-180' : ''}`} />
                                    </button>

                                    {/* User Dropdown */}
                                    {userMenuOpen && (
                                        <div className="absolute right-0 mt-2 w-56 bg-white rounded-xl shadow-xl border border-slate-200 py-2 z-50">
                                            <div className="px-4 py-3 border-b border-slate-100">
                                                <div className="font-semibold text-slate-900">{user.name}</div>
                                                <div className="text-sm text-slate-500">{user.email}</div>
                                            </div>

                                            <Link
                                                to="/profile"
                                                className="flex items-center gap-3 px-4 py-2 text-slate-700 hover:bg-slate-50 transition-colors"
                                            >
                                                <User className="w-4 h-4" />
                                                My Profile
                                            </Link>

                                            <Link
                                                to="/settings"
                                                className="flex items-center gap-3 px-4 py-2 text-slate-700 hover:bg-slate-50 transition-colors"
                                            >
                                                <Settings className="w-4 h-4" />
                                                Settings
                                            </Link>

                                            <Link
                                                to="/my-courses"
                                                className="flex items-center gap-3 px-4 py-2 text-slate-700 hover:bg-slate-50 transition-colors"
                                            >
                                                📚 My Courses
                                            </Link>

                                            <div className="border-t border-slate-100 mt-2 pt-2">
                                                <button
                                                    onClick={() => setIsLoggedIn(false)}
                                                    className="flex items-center gap-3 px-4 py-2 text-red-600 hover:bg-red-50 transition-colors w-full"
                                                >
                                                    <LogOut className="w-4 h-4" />
                                                    Log Out
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </>
                        ) : (
                            <>
                                <Link
                                    to="/login"
                                    className="px-4 py-2 text-slate-700 hover:text-primary-600 transition-colors font-medium"
                                >
                                    Log In
                                </Link>
                                <Link
                                    to="/signup"
                                    className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-lg transition-colors shadow-sm"
                                >
                                    Get Started
                                </Link>
                            </>
                        )}
                    </div>

                    {/* Mobile menu button */}
                    <button
                        className="md:hidden p-2 rounded-lg text-slate-700 hover:bg-slate-100"
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    >
                        {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                    </button>
                </div>

                {/* Mobile Menu */}
                {mobileMenuOpen && (
                    <div className="md:hidden py-4 border-t border-slate-200">
                        <div className="flex flex-col space-y-2">
                            <Link to="/" className="px-4 py-2 text-slate-700 hover:bg-slate-50 rounded-lg font-medium">
                                Home
                            </Link>

                            {/* Mobile Features Submenu */}
                            <div className="px-4 py-2">
                                <button
                                    onClick={() => setFeaturesOpen(!featuresOpen)}
                                    className="flex items-center justify-between w-full text-slate-700 font-medium"
                                >
                                    Features
                                    <ChevronDown className={`w-4 h-4 transition-transform ${featuresOpen ? 'rotate-180' : ''}`} />
                                </button>
                                {featuresOpen && (
                                    <div className="mt-2 ml-4 space-y-2">
                                        {featuresItems.map((feature, index) => (
                                            <a
                                                key={index}
                                                href={feature.href}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="block py-2 text-sm text-slate-600 hover:text-primary-600"
                                            >
                                                {feature.icon} {feature.title}
                                            </a>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Mobile Tools Submenu */}
                            <div className="px-4 py-2">
                                <button
                                    onClick={() => setToolsOpen(!toolsOpen)}
                                    className="flex items-center justify-between w-full text-slate-700 font-medium"
                                >
                                    Tools
                                    <ChevronDown className={`w-4 h-4 transition-transform ${toolsOpen ? 'rotate-180' : ''}`} />
                                </button>
                                {toolsOpen && (
                                    <div className="mt-2 ml-4 space-y-2">
                                        {toolsItems.map((tool, index) => (
                                            <a
                                                key={index}
                                                href={tool.href}
                                                className="block py-2 text-sm text-slate-600 hover:text-primary-600"
                                            >
                                                {tool.icon} {tool.title}
                                            </a>
                                        ))}
                                    </div>
                                )}
                            </div>
                            <div className="hidden lg:flex items-center space-x-4">
                                <Link to="/login" className="text-gray-700 hover:text-indigo-600 font-medium px-4 py-2 transition-colors">
                                    Log in
                                </Link>
                                <Link to="/signup" className="bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2.5 rounded-lg font-medium transition-all shadow-md shadow-indigo-500/20">
                                    Sign up
                                </Link>
                            </div>    <Link to="/community" className="px-4 py-2 text-slate-700 hover:bg-slate-50 rounded-lg font-medium">
                                Community
                            </Link>

                            {isLoggedIn ? (
                                <>
                                    <Link to="/admin" className="px-4 py-2 text-slate-700 hover:bg-slate-50 rounded-lg font-medium">
                                        Admin Panel
                                    </Link>
                                    <Link to="/profile" className="px-4 py-2 text-slate-700 hover:bg-slate-50 rounded-lg font-medium">
                                        My Profile
                                    </Link>
                                    <Link to="/community" className="block px-4 py-2 text-sm text-gray-600 hover:text-indigo-600 hover:bg-gray-50 rounded-lg">
                                        Forums
                                    </Link>
                                    <Link to="/community/members" className="block px-4 py-2 text-sm text-gray-600 hover:text-indigo-600 hover:bg-gray-50 rounded-lg">
                                        MembersList
                                    </Link>
                                    <Link to="/support" className="px-4 py-2 text-slate-700 hover:bg-slate-50 rounded-lg font-medium">
                                        Support
                                    </Link>
                                    <button
                                        onClick={() => setIsLoggedIn(false)}
                                        className="px-4 py-2 text-left text-red-600 hover:bg-red-50 rounded-lg font-medium w-full"
                                    >
                                        Log Out
                                    </button>
                                </>
                            ) : (
                                <div className="pt-4 flex flex-col space-y-3">
                                    <Link
                                        to="/login"
                                        className="px-6 py-2 text-center text-slate-700 hover:bg-slate-100 font-medium rounded-lg transition-colors border border-slate-300"
                                    >
                                        Log In
                                    </Link>
                                    <Link
                                        to="/signup"
                                        className="px-6 py-2 text-center bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-lg transition-colors"
                                    >
                                        Get Started
                                    </Link>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </nav>
        </header>
    );
}
