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
        <footer className="bg-[#0B1121] text-white font-sans border-t border-slate-800">
            {/* CTA Section - Kept inside but styled to blend or stand out cleanly */}
            <div className="bg-primary-600 relative overflow-hidden">
                <div className="absolute inset-0 bg-[url('/assets/images/grid-pattern.png')] opacity-10"></div>
                <div className="container mx-auto px-4 py-16 relative z-10 text-center">
                    <h2 className="text-3xl md:text-4xl font-bold mb-6 text-white">Ready to Accelerate Your Research?</h2>
                    <p className="text-primary-100 text-lg mb-8 max-w-2xl mx-auto">
                        Join thousands of researchers using BioDockify for fast, accurate, and free molecular docking.
                    </p>
                    <Link
                        to="/dock/new"
                        className="inline-flex items-center gap-2 bg-white text-primary-600 px-8 py-4 rounded-full font-bold hover:bg-slate-50 transition-all transform hover:-translate-y-1 shadow-lg"
                    >
                        <span>Start Docking Now</span>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 8l4 4m0 0l-4 4m4-4H3" />
                        </svg>
                    </Link>
                </div>
            </div>

            {/* Main Footer Content */}
            <div className="container mx-auto px-4 py-16 lg:py-24">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-12 lg:gap-8">

                    {/* Brand & Contact (Cols 1-4) */}
                    <div className="lg:col-span-4 space-y-6">
                        <Link to="/" className="inline-block">
                            <div className="flex items-center gap-2 text-white font-bold text-2xl tracking-tight">
                                <span className="text-3xl">🧬</span>
                                <span>Bio<span className="text-primary-500">Dockify</span></span>
                            </div>
                        </Link>
                        <p className="text-slate-400 leading-relaxed text-sm">
                            Cloud-native molecular docking infrastructure.
                            Empowering students and researchers with enterprise-grade tools, completely free.
                        </p>
                        <div className="space-y-4 pt-2">
                            <div className="flex items-start gap-3 text-slate-400 text-sm">
                                <svg className="w-5 h-5 text-primary-500 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                                </svg>
                                <span>123 Innovation Drive,<br />Cambridge, MA 02142</span>
                            </div>
                            <div className="flex items-center gap-3 text-slate-400 text-sm">
                                <svg className="w-5 h-5 text-primary-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                </svg>
                                <span>support@biodockify.com</span>
                            </div>
                        </div>
                    </div>

                    {/* Links Column 1 (Cols 5-6) */}
                    <div className="lg:col-span-2 lg:col-start-6">
                        <h3 className="text-white font-bold mb-6">Company</h3>
                        <ul className="space-y-3 text-sm">
                            <li><Link to="/about" className="text-slate-400 hover:text-primary-400 transition-colors">About Us</Link></li>
                            <li><Link to="/contact" className="text-slate-400 hover:text-primary-400 transition-colors">Contact</Link></li>
                            <li><Link to="/blog" className="text-slate-400 hover:text-primary-400 transition-colors">News</Link></li>
                            <li><Link to="/careers" className="text-slate-400 hover:text-primary-400 transition-colors">Careers</Link></li>
                        </ul>
                    </div>

                    {/* Links Column 2 (Cols 7-8) */}
                    <div className="lg:col-span-2">
                        <h3 className="text-white font-bold mb-6">Resources</h3>
                        <ul className="space-y-3 text-sm">
                            <li><Link to="/dock/new" className="text-slate-400 hover:text-primary-400 transition-colors">Start Docking</Link></li>
                            <li><Link to="/ai-analysis" className="text-slate-400 hover:text-primary-400 transition-colors">AI Analysis</Link></li>
                            <li><Link to="/docs" className="text-slate-400 hover:text-primary-400 transition-colors">Documentation</Link></li>
                            <li><Link to="/faq" className="text-slate-400 hover:text-primary-400 transition-colors">Help Center</Link></li>
                        </ul>
                    </div>

                    {/* Newsletter (Cols 9-12) */}
                    <div className="lg:col-span-4">
                        <h3 className="text-white font-bold mb-6">Newsletter</h3>
                        <p className="text-slate-400 mb-4 text-sm">
                            Subscribe for the latest updates in computational drug discovery.
                        </p>
                        <form onSubmit={handleSubscribe} className="space-y-3">
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="Email Address"
                                required
                                className="w-full bg-slate-800/50 border border-slate-700 rounded-lg py-3 px-4 text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all text-sm"
                            />
                            <button
                                type="submit"
                                className="w-full bg-primary-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-primary-500 transition-colors text-sm"
                            >
                                Subscribe
                            </button>
                        </form>
                    </div>
                </div>
            </div>

            {/* Bottom Bar */}
            <div className="border-t border-slate-800/50 bg-[#0B1121]">
                <div className="container mx-auto px-4 py-8">
                    <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                        <div className="text-slate-500 text-sm">
                            © {new Date().getFullYear()} BioDockify. All rights reserved.
                        </div>

                        <div className="flex gap-6 text-sm">
                            <Link to="/privacy" className="text-slate-500 hover:text-primary-400 transition-colors">Privacy Policy</Link>
                            <Link to="/terms" className="text-slate-500 hover:text-primary-400 transition-colors">Terms of Service</Link>
                        </div>

                        <div className="flex items-center gap-4">
                            <a href="#" className="w-8 h-8 rounded-full bg-slate-800/50 flex items-center justify-center text-slate-400 hover:bg-primary-600 hover:text-white transition-all">
                                <span className="sr-only">Twitter</span>
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5a4.5 4.5 0 00-.08-.83A7.72 7.72 0 0023 3z"></path></svg>
                            </a>
                            <a href="#" className="w-8 h-8 rounded-full bg-slate-800/50 flex items-center justify-center text-slate-400 hover:bg-primary-600 hover:text-white transition-all">
                                <span className="sr-only">GitHub</span>
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd"></path></svg>
                            </a>
                            <a href="#" className="w-8 h-8 rounded-full bg-slate-800/50 flex items-center justify-center text-slate-400 hover:bg-primary-600 hover:text-white transition-all">
                                <span className="sr-only">LinkedIn</span>
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"></path></svg>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    )
}
