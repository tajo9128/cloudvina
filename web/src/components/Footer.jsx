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
            {/* CTA Section */}
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

            <div className="container mx-auto px-4 py-12">
                {/* ROW 1: Logo + Copyright | Contact Info */}
                {/* ROW 1: Logo (1/3) + Address (2/3) */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 pb-12 border-b border-slate-800/50">
                    {/* Logo Section (1/3) */}
                    <div className="lg:col-span-1">
                        <Link to="/" className="inline-block mb-4">
                            <div className="flex items-center gap-2 text-white font-bold text-2xl tracking-tight">
                                <span className="text-3xl">🧬</span>
                                <span>Bio<span className="text-primary-500">Dockify</span></span>
                            </div>
                        </Link>
                        <p className="text-slate-400 text-sm leading-relaxed">
                            Accelerating drug discovery with cloud-powered molecular docking. Fast, accurate, and accessible for every lab.
                        </p>
                    </div>

                    {/* Address/Contact Section (2/3) */}
                    <div className="lg:col-span-2 flex flex-col md:flex-row items-start md:items-center justify-start md:justify-end gap-8 h-full">
                        <div className="flex items-center gap-4 text-slate-300">
                            <div className="w-12 h-12 rounded-lg bg-slate-800/50 flex items-center justify-center text-primary-500 shrink-0">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                </svg>
                            </div>
                            <div>
                                <p className="text-xs text-slate-500 uppercase font-bold tracking-wider mb-1">Email Us</p>
                                <p className="font-semibold">biodockify@hotmail.com</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-4 text-slate-300">
                            <div className="w-12 h-12 rounded-lg bg-slate-800/50 flex items-center justify-center text-primary-500 shrink-0">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                                </svg>
                            </div>
                            <div>
                                <p className="text-xs text-slate-500 uppercase font-bold tracking-wider mb-1">Visit Us</p>
                                <p className="font-semibold">Bala Nagar, Hyderabad, Telangana, India</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ROW 2: Newsletter + Social | Resources | Products | Company */}
                {/* ROW 2: Newsletter (1/3) + Footer Menu (2/3) */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 py-12 border-b border-slate-800/50">
                    {/* Newsletter (1/3) */}
                    <div className="lg:col-span-1">
                        <h3 className="text-white font-bold mb-6 text-lg">Subscribe Newsletter</h3>
                        <p className="text-slate-400 text-sm mb-6">Get the latest updates on new features and drug discovery insights.</p>
                        <form onSubmit={handleSubscribe} className="space-y-4">
                            <div className="relative">
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="Your email address"
                                    required
                                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg py-3 px-4 text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-all text-sm"
                                />
                                <button
                                    type="submit"
                                    className="absolute right-1 top-1 bottom-1 bg-primary-600 text-white px-4 rounded-md hover:bg-primary-500 transition-colors flex items-center justify-center"
                                    aria-label="Subscribe"
                                >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                                    </svg>
                                </button>
                            </div>
                        </form>
                    </div>

                    {/* Footer Menu (2/3) */}
                    <div className="lg:col-span-2">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 h-full">
                            {/* Resources */}
                            <div>
                                <h3 className="text-white font-bold mb-6 text-base">Resources</h3>
                                <ul className="space-y-3 text-sm">
                                    <li><Link to="/dock/new" className="text-slate-400 hover:text-primary-400 transition-colors">Start Docking</Link></li>
                                    <li><Link to="/ai-analysis" className="text-slate-400 hover:text-primary-400 transition-colors">AI Analysis</Link></li>
                                    <li><Link to="/docs" className="text-slate-400 hover:text-primary-400 transition-colors">Documentation</Link></li>
                                    <li><Link to="/faq" className="text-slate-400 hover:text-primary-400 transition-colors">Help Center</Link></li>
                                </ul>
                            </div>

                            {/* Products */}
                            <div>
                                <h3 className="text-white font-bold mb-6 text-base">Products</h3>
                                <ul className="space-y-3 text-sm">
                                    <li><Link to="/#features" className="text-slate-400 hover:text-primary-400 transition-colors">AutoDock Vina</Link></li>
                                    <li><Link to="/ai-analysis" className="text-slate-400 hover:text-primary-400 transition-colors">AI Explainer</Link></li>
                                    <li><Link to="/#pricing" className="text-slate-400 hover:text-primary-400 transition-colors">Cloud Platform</Link></li>
                                    <li><Link to="/dashboard" className="text-slate-400 hover:text-primary-400 transition-colors">Dashboard</Link></li>
                                </ul>
                            </div>

                            {/* Company */}
                            <div>
                                <h3 className="text-white font-bold mb-6 text-base">Company</h3>
                                <ul className="space-y-3 text-sm">
                                    <li><Link to="/about" className="text-slate-400 hover:text-primary-400 transition-colors">About Us</Link></li>
                                    <li><Link to="/contact" className="text-slate-400 hover:text-primary-400 transition-colors">Contact</Link></li>
                                    <li><Link to="/blog" className="text-slate-400 hover:text-primary-400 transition-colors">News</Link></li>
                                    <li><Link to="/careers" className="text-slate-400 hover:text-primary-400 transition-colors">Careers</Link></li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ROW 3: Copyright | Legal Links | Social Icons */}
                <div className="py-6">
                    <div className="flex flex-col md:flex-row justify-between items-center gap-6">
                        {/* Copyright */}
                        <p className="text-slate-500 text-sm">
                            © {new Date().getFullYear()} BioDockify. All rights reserved.
                        </p>

                        {/* Legal Links */}
                        <div className="flex items-center gap-6 text-sm">
                            <Link to="/privacy" className="text-slate-500 hover:text-primary-400 transition-colors">Privacy Policy</Link>
                            <Link to="/terms" className="text-slate-500 hover:text-primary-400 transition-colors">Terms of Service</Link>
                        </div>

                        {/* Social Icons */}
                        <div className="flex items-center gap-2">
                            <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="w-8 h-8 rounded-lg bg-slate-800/50 flex items-center justify-center text-slate-400 hover:bg-primary-600 hover:text-white transition-all">
                                <span className="sr-only">Instagram</span>
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4z" /></svg>
                            </a>
                            <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="w-8 h-8 rounded-lg bg-slate-800/50 flex items-center justify-center text-slate-400 hover:bg-primary-600 hover:text-white transition-all">
                                <span className="sr-only">Facebook</span>
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" /></svg>
                            </a>
                            <a href="https://youtube.com" target="_blank" rel="noopener noreferrer" className="w-8 h-8 rounded-lg bg-slate-800/50 flex items-center justify-center text-slate-400 hover:bg-primary-600 hover:text-white transition-all">
                                <span className="sr-only">YouTube</span>
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" /></svg>
                            </a>
                            <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="w-8 h-8 rounded-lg bg-slate-800/50 flex items-center justify-center text-slate-400 hover:bg-primary-600 hover:text-white transition-all">
                                <span className="sr-only">LinkedIn</span>
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" /></svg>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    )
}
