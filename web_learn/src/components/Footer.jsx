import React from 'react';
import { Link } from 'react-router-dom';
import { Github, Twitter, Linkedin, Mail } from 'lucide-react';

export default function Footer() {
    return (
        <footer className="bg-slate-950 text-slate-100">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
                    {/* Brand Column */}
                    <div className="md:col-span-1">
                        <Link to="/" className="flex items-center space-x-2 mb-4">
                            <img
                                src="/brand/logo.svg"
                                alt="BioDockify"
                                className="h-16 w-auto brightness-0 invert opacity-90"
                            />
                        </Link>
                        <p className="text-slate-400 text-sm leading-relaxed">
                            Master molecular docking with comprehensive courses, tutorials, and community support.
                        </p>
                        <div className="flex items-center space-x-4 mt-6">
                            <a href="https://github.com/tajo9128/cloudvina" target="_blank" rel="noopener noreferrer"
                                className="text-slate-400 hover:text-white transition-colors">
                                <Github className="w-5 h-5" />
                            </a>
                            <a href="#" className="text-slate-400 hover:text-white transition-colors">
                                <Twitter className="w-5 h-5" />
                            </a>
                            <a href="#" className="text-slate-400 hover:text-white transition-colors">
                                <Linkedin className="w-5 h-5" />
                            </a>
                        </div>
                    </div>

                    {/* Platform Links */}
                    <div>
                        <h3 className="font-semibold text-white mb-4">Platform</h3>
                        <ul className="space-y-3">
                            <li>
                                <Link to="/courses" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Courses
                                </Link>
                            </li>
                            <li>
                                <Link to="/blog" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Blog
                                </Link>
                            </li>
                            <li>
                                <Link to="/community" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Community
                                </Link>
                            </li>
                            <li>
                                <a href="https://www.biodockify.com" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Main Platform
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
                                <Link to="/tutorials" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    Tutorials
                                </Link>
                            </li>
                            <li>
                                <Link to="/faq" className="text-slate-400 hover:text-white transition-colors text-sm">
                                    FAQ
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
                        <h3 className="font-semibold text-white mb-4">Stay Updated</h3>
                        <p className="text-slate-400 text-sm mb-4">
                            Get the latest courses and resources delivered to your inbox.
                        </p>
                        <form className="space-y-3">
                            <input
                                type="email"
                                placeholder="Your email"
                                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-colors"
                            />
                            <button className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-lg transition-colors">
                                Subscribe
                            </button>
                        </form>
                    </div>
                </div>

                {/* Bottom Bar */}
                <div className="pt-8 border-t border-slate-800">
                    <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
                        <p className="text-slate-500 text-sm">
                            © 2024 BioDockify. All rights reserved.
                        </p>
                        <div className="flex items-center space-x-6">
                            <Link to="/privacy" className="text-slate-500 hover:text-white transition-colors text-sm">
                                Privacy Policy
                            </Link>
                            <Link to="/terms" className="text-slate-500 hover:text-white transition-colors text-sm">
                                Terms of Service
                            </Link>
                            <Link to="/contact" className="text-slate-500 hover:text-white transition-colors text-sm">
                                Contact
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    );
}
