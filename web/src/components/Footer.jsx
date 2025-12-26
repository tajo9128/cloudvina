import React from 'react';
import { Link } from 'react-router-dom';

export default function Footer() {
    return (
        <footer className="bg-slate-900 border-t border-slate-800 pt-16 pb-8 text-slate-400">
            <div className="container mx-auto px-4">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-12">
                    {/* Brand */}
                    <div className="space-y-4">
                        <Link to="/" className="flex items-center space-x-2 group">
                            <img
                                src="/brand/logo.svg"
                                alt="BioDockify"
                                className="h-12 w-auto brightness-0 invert opacity-90 group-hover:opacity-100 transition-opacity duration-300"
                            />
                        </Link>
                        <p className="text-sm leading-relaxed text-slate-500">
                            Accelerating drug discovery with AI-powered molecular docking and analysis tools.
                            Zero-cost, open-source, and accessible to everyone.
                        </p>
                    </div>

                    {/* Quick Links */}
                    <div>
                        <h3 className="text-white font-semibold mb-4">Platform</h3>
                        <ul className="space-y-2 text-sm">
                            <li><Link to="/discovery" className="hover:text-primary-400 transition-colors">Drug Discovery</Link></li>
                            <li><Link to="/ai-tools" className="hover:text-primary-400 transition-colors">AI Suite</Link></li>
                            <li><Link to="/analysis" className="hover:text-primary-400 transition-colors">Analysis</Link></li>
                            <li><Link to="/pricing" className="hover:text-primary-400 transition-colors">Pricing (Free)</Link></li>
                        </ul>
                    </div>

                    {/* Resources */}
                    <div>
                        <h3 className="text-white font-semibold mb-4">Resources</h3>
                        <ul className="space-y-2 text-sm">
                            <li><Link to="/docs" className="hover:text-primary-400 transition-colors">Documentation</Link></li>
                            <li><Link to="/tutorials" className="hover:text-primary-400 transition-colors">Tutorials</Link></li>
                            <li><Link to="/support" className="hover:text-primary-400 transition-colors">Support</Link></li>
                            <li><a href="https://github.com/tajo9128/cloudvina" target="_blank" rel="noreferrer" className="hover:text-primary-400 transition-colors">GitHub</a></li>
                        </ul>
                    </div>

                    {/* Newsletter */}
                    <div>
                        <h3 className="text-white font-semibold mb-4">Stay Updated</h3>
                        <form className="space-y-2">
                            <div className="relative">
                                <input
                                    type="email"
                                    placeholder="Enter your email"
                                    className="w-full bg-slate-800 border border-slate-700 rounded-lg py-2 px-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500 transition-all"
                                />
                                <button
                                    type="submit"
                                    className="absolute right-1 top-1 bg-primary-600 hover:bg-primary-500 text-white p-1.5 rounded-md transition-colors"
                                >
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                                    </svg>
                                </button>
                            </div>
                            <p className="text-xs text-slate-500">
                                Weekly updates on new models and tools. No spam.
                            </p>
                        </form>
                    </div>
                </div>

                <div className="border-t border-slate-800 pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-xs">
                    <p>© 2025 BioDockify. Open Source (MIT License).</p>
                    <div className="flex gap-6">
                        <Link to="/privacy" className="hover:text-white transition-colors">Privacy Policy</Link>
                        <Link to="/refund-policy" className="hover:text-white transition-colors">Refund Policy</Link>
                        <Link to="/terms" className="hover:text-white transition-colors">Terms of Service</Link>
                    </div>
                </div>
            </div>
        </footer>
    );
}
