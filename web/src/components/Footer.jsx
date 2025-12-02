import { Link } from 'react-router-dom'
import { useState } from 'react'

export default function Footer() {
    const [email, setEmail] = useState('')

    const handleSubscribe = (e) => {
        e.preventDefault()
        // TODO: Implement newsletter subscription
        console.log('Subscribe:', email)
        setEmail('')
    }

    return (
        <footer className="bg-[#0f172a] text-slate-300 border-t border-slate-800">
            {/* Main Footer Content */}
            <div className="container mx-auto px-4 py-20">
                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-12 lg:gap-8">

                    {/* Column 1: Brand & About */}
                    <div className="space-y-6">
                        <Link to="/" className="inline-block">
                            <div className="flex items-center gap-2 text-white font-bold text-2xl tracking-tight">
                                <span className="text-3xl">🧬</span>
                                Bio<span className="text-primary-500">Dockify</span>
                            </div>
                        </Link>
                        <p className="text-slate-400 text-sm leading-relaxed">
                            Accelerating drug discovery with cloud-native molecular docking. Powered by AutoDock Vina on AWS infrastructure for scalable, publication-ready results.
                        </p>
                        <div className="pt-4">
                            <div className="flex items-center gap-3 text-sm text-slate-400 mb-2">
                                <svg className="w-4 h-4 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"></path>
                                </svg>
                                <span>+1 (617) 789-1234</span>
                            </div>
                            <div className="flex items-center gap-3 text-sm text-slate-400">
                                <svg className="w-4 h-4 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
                                </svg>
                                <span>support@biodockify.com</span>
                            </div>
                        </div>
                    </div>

                    {/* Column 2: Useful Links */}
                    <div>
                        <h3 className="text-white font-bold text-lg mb-6">Useful Links</h3>
                        <ul className="space-y-3">
                            <li>
                                <Link to="/about" className="text-slate-400 hover:text-primary-500 transition-colors text-sm flex items-center gap-2 group">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-primary-500 transition-colors"></span>
                                    About Us
                                </Link>
                            </li>
                            <li>
                                <Link to="/dock/new" className="text-slate-400 hover:text-primary-500 transition-colors text-sm flex items-center gap-2 group">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-primary-500 transition-colors"></span>
                                    Start Docking
                                </Link>
                            </li>
                            <li>
                                <Link to="/ai-analysis" className="text-slate-400 hover:text-primary-500 transition-colors text-sm flex items-center gap-2 group">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-primary-500 transition-colors"></span>
                                    AI Analysis
                                </Link>
                            </li>
                            <li>
                                <Link to="/blog" className="text-slate-400 hover:text-primary-500 transition-colors text-sm flex items-center gap-2 group">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-primary-500 transition-colors"></span>
                                    Latest News
                                </Link>
                            </li>
                            <li>
                                <Link to="/contact" className="text-slate-400 hover:text-primary-500 transition-colors text-sm flex items-center gap-2 group">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-primary-500 transition-colors"></span>
                                    Contact Us
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* Column 3: Legal & Support */}
                    <div>
                        <h3 className="text-white font-bold text-lg mb-6">Legal & Support</h3>
                        <ul className="space-y-3">
                            <li>
                                <Link to="/privacy" className="text-slate-400 hover:text-primary-500 transition-colors text-sm flex items-center gap-2 group">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-primary-500 transition-colors"></span>
                                    Privacy Policy
                                </Link>
                            </li>
                            <li>
                                <Link to="/terms" className="text-slate-400 hover:text-primary-500 transition-colors text-sm flex items-center gap-2 group">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-primary-500 transition-colors"></span>
                                    Terms & Conditions
                                </Link>
                            </li>
                            <li>
                                <Link to="/faq" className="text-slate-400 hover:text-primary-500 transition-colors text-sm flex items-center gap-2 group">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-primary-500 transition-colors"></span>
                                    FAQ
                                </Link>
                            </li>
                            <li>
                                <Link to="/support" className="text-slate-400 hover:text-primary-500 transition-colors text-sm flex items-center gap-2 group">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-primary-500 transition-colors"></span>
                                    Help Center
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* Column 4: Newsletter */}
                    <div>
                        <h3 className="text-white font-bold text-lg mb-6">Newsletter</h3>
                        <p className="text-slate-400 text-sm mb-4 leading-relaxed">
                            Subscribe to our newsletter to get the latest updates and news.
                        </p>
                        <form onSubmit={handleSubscribe} className="space-y-3">
                            <div className="relative">
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="Email Address"
                                    required
                                    className="w-full bg-slate-900/50 border border-slate-700 rounded-lg py-3 px-4 text-white text-sm focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all placeholder-slate-500"
                                />
                            </div>
                            <button
                                type="submit"
                                className="w-full bg-primary-600 hover:bg-primary-500 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-300 hover:shadow-lg hover:shadow-primary-500/20 text-sm uppercase tracking-wide flex items-center justify-center gap-2"
                            >
                                <span>Subscribe</span>
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
                                </svg>
                            </button>
                        </form>
                    </div>
                </div>
            </div>

            {/* Bottom Bar */}
            <div className="border-t border-slate-800 bg-[#0b1120]">
                <div className="container mx-auto px-4 py-6">
                    <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                        <div className="text-sm text-slate-500">
                            © {new Date().getFullYear()} BioDockify. All rights reserved.
                        </div>

                        {/* Social Media Icons */}
                        <div className="flex items-center gap-4">
                            <a href="https://twitter.com/biodockify" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white transition-colors">
                                <span className="sr-only">Twitter</span>
                                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5a4.5 4.5 0 00-.08-.83A7.72 7.72 0 0023 3z"></path>
                                </svg>
                            </a>
                            <a href="https://github.com/biodockify" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white transition-colors">
                                <span className="sr-only">GitHub</span>
                                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd"></path>
                                </svg>
                            </a>
                            <a href="https://linkedin.com/company/biodockify" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white transition-colors">
                                <span className="sr-only">LinkedIn</span>
                                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"></path>
                                </svg>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    )
}
