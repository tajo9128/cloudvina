import React from 'react';
import { Link } from 'react-router-dom';
import { Github, Twitter, Linkedin, Mail } from 'lucide-react';

export default function Footer() {
    return (
        <footer className="bg-slate-950 text-slate-100 font-sans">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
                    {/* Brand Column */}
                    <div className="md:col-span-1">
                        <Link to="/" className="flex items-center space-x-2 mb-4">
                            <img src="/brand/logo.svg" alt="BioDockify Logo" className="h-8 w-auto" />
                            <span className="text-xl font-bold font-display">
                                <img src="/brand/logo.svg" alt="BioDockify Logo" className="h-8 w-auto" />
                                <span className="text-primary-500">Dockify</span>
                            </span>
                        </Link>
                        <p className="text-slate-400 text-sm leading-relaxed">
                            AI-Powered Molecular Docking & QSAR Prediction Platform.
                        </p>
                        <div className="flex items-center space-x-4 mt-6">
                            <a href="https://github.com/tajo9128/cloudvina" target="_blank" rel="noopener noreferrer"
                                className="text-slate-400 hover:text-white transition-colors">
                                <Github className="w-5 h-5" />
                            </a>
                            <a href="#" className="text-slate-400 hover:text-white transition-colors">
                                <Twitter className="w-5 h-5" />
                            </a>
                        </div>
                    </div>

                    {/* Platform Links */}
                    <div>
                        <h3 className="font-semibold text-white mb-4">Platform</h3>
                        <ul className="space-y-3">
                            <li>
                                <Link to="/" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Dashboard
                                </Link>
                            </li>
                            <li>
                                <a href="https://learn.biodockify.com/courses" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Courses
                                </a>
                            </li>
                            <li>
                                <a href="https://learn.biodockify.com/blog" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Blog
                                </a>
                            </li>
                        </ul>
                    </div>

                    {/* Resources Links */}
                    <div>
                        <h3 className="font-semibold text-white mb-4">Resources</h3>
                        <ul className="space-y-3">
                            <li>
                                <Link to="/docs" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Documentation
                                </Link>
                            </li>
                            <li>
                                <a href="https://www.biodockify.com/pricing" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Pricing
                                </a>
                            </li>
                        </ul>
                    </div>

                    {/* Newsletter */}
                    <div>
                        <h3 className="font-semibold text-white mb-4">Stay Connected</h3>
                        <p className="text-slate-400 text-sm mb-4">
                            Join our research community.
                        </p>
                    </div>
                </div>

                {/* Bottom Bar */}
                <div className="pt-8 border-t border-slate-800">
                    <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
                        <p className="text-slate-500 text-sm">
                            © 2024 BioDockify. All rights reserved.
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    );
}
