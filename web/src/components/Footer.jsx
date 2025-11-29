import { Link } from 'react-router-dom'

export default function Footer() {
    return (
        <footer className="bg-deep-navy-900 text-white py-16 border-t border-white/5 relative overflow-hidden">
            {/* Background Glow */}
            <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[100px] pointer-events-none"></div>
            <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-600/10 rounded-full blur-[100px] pointer-events-none"></div>

            <div className="container mx-auto px-4 relative z-10">
                <div className="grid md:grid-cols-4 gap-12 mb-12">
                    <div>
                        <Link to="/" className="flex items-center space-x-2 mb-6 group">
                            <div className="text-3xl transform group-hover:scale-110 transition-transform duration-300 drop-shadow-[0_0_8px_rgba(0,217,255,0.5)]">🧬</div>
                            <h2 className="text-2xl font-extrabold tracking-tight">Bio<span className="text-cyan-400">Dockify</span></h2>
                        </Link>
                        <p className="text-blue-200/80 text-sm leading-relaxed mb-6">
                            Democratizing drug discovery with cloud-native molecular docking tools. Accelerate your research with our powerful, accessible platform.
                        </p>
                        <div className="flex space-x-4">
                            {/* Social Icons Placeholders */}
                            <a href="#" className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center hover:bg-cyan-500 hover:text-white transition-all duration-300">
                                <span className="sr-only">Twitter</span>
                                🐦
                            </a>
                            <a href="#" className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center hover:bg-cyan-500 hover:text-white transition-all duration-300">
                                <span className="sr-only">GitHub</span>
                                🐙
                            </a>
                            <a href="#" className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center hover:bg-cyan-500 hover:text-white transition-all duration-300">
                                <span className="sr-only">LinkedIn</span>
                                💼
                            </a>
                        </div>
                    </div>

                    <div>
                        <h3 className="font-bold text-white mb-6 text-lg">Product</h3>
                        <ul className="space-y-3 text-blue-200/70 text-sm">
                            <li><Link to="/#features" className="hover:text-cyan-400 transition-colors flex items-center gap-2"><span className="w-1 h-1 bg-cyan-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"></span>Features</Link></li>
                            <li><Link to="/#pricing" className="hover:text-cyan-400 transition-colors">Pricing</Link></li>
                            <li><Link to="/tools/converter" className="hover:text-cyan-400 transition-colors">SDF Converter</Link></li>
                            <li><Link to="/admin" className="hover:text-cyan-400 transition-colors">Admin</Link></li>
                        </ul>
                    </div>

                    <div>
                        <h3 className="font-bold text-white mb-6 text-lg">Company</h3>
                        <ul className="space-y-3 text-blue-200/70 text-sm">
                            <li><Link to="/about" className="hover:text-cyan-400 transition-colors">About Us</Link></li>
                            <li><Link to="/blog" className="hover:text-cyan-400 transition-colors">Blog</Link></li>
                            <li><Link to="/contact" className="hover:text-cyan-400 transition-colors">Contact</Link></li>
                            <li><Link to="/careers" className="hover:text-cyan-400 transition-colors">Careers</Link></li>
                        </ul>
                    </div>

                    <div>
                        <h3 className="font-bold text-white mb-6 text-lg">Legal</h3>
                        <ul className="space-y-3 text-blue-200/70 text-sm">
                            <li><Link to="/privacy" className="hover:text-cyan-400 transition-colors">Privacy Policy</Link></li>
                            <li><Link to="/terms" className="hover:text-cyan-400 transition-colors">Terms of Service</Link></li>
                            <li><Link to="/cookies" className="hover:text-cyan-400 transition-colors">Cookie Policy</Link></li>
                        </ul>
                    </div>
                </div>

                <div className="border-t border-white/10 pt-8 flex flex-col md:flex-row justify-between items-center text-blue-200/50 text-sm">
                    <p>© {new Date().getFullYear()} BioDockify. All rights reserved.</p>
                    <p className="mt-2 md:mt-0">Designed for Science 🧪</p>
                </div>
            </div>
        </footer>
    )
}
