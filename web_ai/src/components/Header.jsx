import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Menu, X, ChevronDown, User, Settings, LogOut, HelpCircle, Bell } from 'lucide-react';

export default function Header() {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [featuresOpen, setFeaturesOpen] = useState(false);
    const [toolsOpen, setToolsOpen] = useState(false);
    const [userMenuOpen, setUserMenuOpen] = useState(false);

    // Mock user state - replace with real auth context later
    // For now, we reuse the same logic as web_learn for consistency
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
        <header className="sticky top-0 z-50 bg-white border-b border-slate-200 shadow-sm font-sans">
            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <div className="flex items-center">
                        <Link to="/" className="flex items-center gap-3">
                            <img src="/brand/logo.svg" alt="BioDockify" className="h-10 w-auto" />
                            <span className="px-2 py-0.5 text-xs font-medium bg-primary-100 text-primary-700 rounded border border-primary-200">
                                AI Hub
                            </span>
                        </Link>
                    </div>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center space-x-1">
                        <Link
                            to="/"
                            className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium"
                        >
                            Dashboard
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

                        <a
                            href="https://learn.biodockify.com/courses"
                            className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium"
                        >
                            Courses
                        </a>

                        <a
                            href="https://learn.biodockify.com/blog"
                            className="px-4 py-2 text-slate-700 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors font-medium"
                        >
                            Blog
                        </a>
                    </div>

                    {/* Right Side: Notifications + User Menu */}
                    <div className="hidden md:flex items-center space-x-3">
                        {/* Always show logged in UI for this mock port or connect to real auth */}
                        {/* For now, just a placeholder login button if we want to be strict, but user asked to COPY the header so I'll keep the structure */}
                        <button className="relative p-2 text-slate-600 hover:text-primary-600 hover:bg-slate-50 rounded-lg transition-colors">
                            <Bell className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Mobile menu button */}
                    <button
                        className="md:hidden p-2 rounded-lg text-slate-700 hover:bg-slate-100"
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    >
                        {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                    </button>
                </div>
            </nav>
        </header>
    );
}
