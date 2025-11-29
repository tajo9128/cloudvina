import { Link } from 'react-router-dom'

export default function HomePage() {
    return (
        <div className="bg-blue-mesh min-h-screen">
            {/* Hero Section */}
            {/* Hero Section */}
            <section className="pt-20 pb-24 relative overflow-hidden">
                {/* Animated Background Elements */}
                <div className="absolute inset-0 pointer-events-none">
                    <div className="absolute top-20 left-10 w-72 h-72 bg-blue-600/20 rounded-full filter blur-3xl animate-float"></div>
                    <div className="absolute bottom-20 right-10 w-96 h-96 bg-cyan-500/20 rounded-full filter blur-3xl animate-float" style={{ animationDelay: '2s' }}></div>
                </div>

                <div className="container mx-auto px-4 text-center relative z-10">
                    <div className="inline-block bg-white/5 backdrop-blur-sm border border-cyan-400/30 text-cyan-300 px-6 py-2 rounded-full text-sm font-bold mb-8 shadow-[0_0_20px_rgba(0,217,255,0.2)]">
                        🎉 Get 130 FREE Credits on Signup!
                    </div>
                    <h1 className="text-5xl md:text-7xl font-extrabold text-white mb-6 leading-tight tracking-tight">
                        Molecular Docking <br />
                        <span className="text-gradient-cyan">for Indian Researchers</span>
                    </h1>
                    <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto font-light">
                        Start with <strong className="text-cyan-400">100 bonus credits</strong> + <strong className="text-cyan-400">30 monthly credits</strong> for free.
                        No credit card required.
                    </p>
                    <p className="text-lg text-blue-200/60 mb-10 max-w-2xl mx-auto">
                        High-performance AutoDock Vina platform powered by AWS. Affordable pricing in ₹. Built for students, researchers, and institutions.
                    </p>
                    <div className="flex flex-wrap justify-center gap-4">
                        <Link to="/login" className="btn-cyan text-lg px-10 py-4 rounded-xl font-bold">
                            Start Free with 130 Credits →
                        </Link>
                        <a href="#pricing" className="btn-cyan-outline px-10 py-4 rounded-xl text-lg font-bold">
                            View Pricing
                        </a>
                    </div>
                </div>
            </section>

            {/* Free Tools Section */}
            <section className="py-20 relative">
                <div className="container mx-auto px-4 relative z-10">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-white mb-4">Free Research Tools</h2>
                        <p className="text-blue-200 max-w-2xl mx-auto font-light">
                            Essential utilities for computational chemistry, available for free.
                        </p>
                    </div>
                    <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8">
                        <div className="glass-modern p-8 rounded-2xl hover:border-cyan-400/50 transition-colors group">
                            <div className="text-5xl mb-6 bg-blue-900/50 w-20 h-20 rounded-2xl flex items-center justify-center border border-blue-700/50">🧪</div>
                            <h3 className="text-2xl font-bold text-white mb-3">SDF to PDBQT Converter</h3>
                            <p className="text-blue-200/70 mb-8 leading-relaxed">
                                Convert your ligand files from SDF format to PDBQT format instantly.
                                Optimized for AutoDock Vina compatibility.
                            </p>
                            <Link to="/tools/converter" className="text-cyan-400 font-bold hover:text-cyan-300 flex items-center group-hover:translate-x-1 transition-transform">
                                Try Converter <span className="ml-2">→</span>
                            </Link>
                        </div>
                        <div className="glass-modern p-8 rounded-2xl opacity-70 grayscale hover:grayscale-0 hover:opacity-100 transition-all duration-500">
                            <div className="text-5xl mb-6 bg-blue-900/50 w-20 h-20 rounded-2xl flex items-center justify-center border border-blue-700/50">📊</div>
                            <h3 className="text-2xl font-bold text-white mb-3">Results Analyzer</h3>
                            <p className="text-blue-200/70 mb-8 leading-relaxed">
                                Visualize and analyze your docking results with advanced 3D rendering.
                                (Coming Soon)
                            </p>
                            <span className="text-blue-400 font-bold cursor-not-allowed border border-blue-400/30 px-3 py-1 rounded-lg text-sm">
                                Coming Soon
                            </span>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section id="features" className="py-20 relative">
                <div className="container mx-auto px-4 relative z-10">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-white mb-4">Why BioDockify?</h2>
                        <p className="text-blue-200 max-w-2xl mx-auto font-light">
                            Built for researchers who need speed, reliability, and flexibility.
                        </p>
                    </div>
                    <div className="grid md:grid-cols-3 gap-8">
                        <div className="glass-modern p-8 rounded-2xl hover:-translate-y-2 transition-transform duration-300">
                            <div className="text-5xl mb-6">🎁</div>
                            <h3 className="text-2xl font-bold text-white mb-3">Generous Free Tier</h3>
                            <p className="text-blue-100/80 leading-relaxed">
                                <strong className="text-cyan-400">100 bonus credits</strong> on signup (30-day expiry) + <strong className="text-cyan-400">30 monthly credits</strong> (recurring).
                                Start with 130 credits for free!
                            </p>
                        </div>
                        <div className="glass-modern p-8 rounded-2xl hover:-translate-y-2 transition-transform duration-300">
                            <div className="text-5xl mb-6">⚡</div>
                            <h3 className="text-2xl font-bold text-white mb-3">Smart Rate Limiting</h3>
                            <p className="text-blue-100/80 leading-relaxed">
                                Free users: <strong className="text-cyan-400">3 jobs/day</strong> for first month, then 1/day.
                                Paid users: <strong className="text-cyan-400">unlimited daily jobs</strong> - use credits anytime!
                            </p>
                        </div>
                        <div className="glass-modern p-8 rounded-2xl hover:-translate-y-2 transition-transform duration-300">
                            <div className="text-5xl mb-6">💎</div>
                            <h3 className="text-2xl font-bold text-white mb-3">Credits Never Expire</h3>
                            <p className="text-blue-100/80 leading-relaxed">
                                Paid plan credits <strong className="text-cyan-400">never expire</strong>. Use them at your own pace.
                                Auto-downgrade to free tier when exhausted.
                            </p>
                        </div>
                        <div className="glass-modern p-8 rounded-2xl hover:-translate-y-2 transition-transform duration-300">
                            <div className="text-5xl mb-6">🔒</div>
                            <h3 className="text-2xl font-bold text-white mb-3">Secure & Verified</h3>
                            <p className="text-blue-100/80 leading-relaxed">
                                Email verification required for all users.
                                Your data is encrypted and containers destroyed after use.
                            </p>
                        </div>
                        <div className="glass-modern p-8 rounded-2xl hover:-translate-y-2 transition-transform duration-300">
                            <div className="text-5xl mb-6">🇮🇳</div>
                            <h3 className="text-2xl font-bold text-white mb-3">Built for India</h3>
                            <p className="text-blue-100/80 leading-relaxed">
                                Affordable pricing in <strong className="text-cyan-400">Indian Rupees</strong> (₹).
                                Student plans starting at just ₹99/month. Special rates for institutions.
                            </p>
                        </div>
                        <div className="glass-modern p-8 rounded-2xl hover:-translate-y-2 transition-transform duration-300">
                            <div className="text-5xl mb-6">⚙️</div>
                            <h3 className="text-2xl font-bold text-white mb-3">AWS Powered</h3>
                            <p className="text-blue-100/80 leading-relaxed">
                                Powered by AWS Fargate. Run hundreds of docking jobs in parallel without waiting for queues.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Pricing Section */}
            <section id="pricing" className="py-20 relative">
                <div className="container mx-auto px-4 relative z-10">
                    <div className="text-center mb-16">
                        <h2 className="text-5xl font-bold text-white mb-4">Simple, Transparent Pricing</h2>
                        <p className="text-xl text-blue-200 font-light">Start free, scale as you grow. Perfect for Indian students and researchers.</p>
                    </div>
                    <div className="max-w-7xl mx-auto grid md:grid-cols-4 gap-6">
                        {/* Free Tier */}
                        <div className="glass-modern p-8 rounded-2xl hover:border-cyan-400/30 transition-all">
                            <div className="mb-6">
                                <div className="text-cyan-400 font-bold text-sm mb-2 tracking-wider">FREE FOREVER</div>
                                <h3 className="text-2xl font-bold text-white mb-2">Free Plan</h3>
                                <div className="flex items-baseline mb-4">
                                    <span className="text-5xl font-bold text-white">₹0</span>
                                </div>
                                <p className="text-gray-600 text-sm">Perfect for students getting started</p>
                            </div>
                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span><strong>100 bonus credits</strong> (30-day signup bonus)</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span><strong>30 credits/month</strong> (recurring, 1/day)</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span><strong>3 jobs/day</strong> (first month)</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>1 job/day (after 1st month)</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>Email verification</span>
                                </li>
                            </ul>
                            <Link to="/login" className="btn-cyan-outline block w-full py-3 px-6 text-center rounded-xl">
                                Start Free
                            </Link>
                        </div>

                        {/* Student Plan */}
                        <div className="glass-modern p-8 rounded-2xl border-2 border-cyan-400 relative shadow-[0_0_30px_rgba(0,217,255,0.15)] transform scale-105 z-10">
                            <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-cyan-400 text-blue-900 text-xs font-bold px-4 py-1 rounded-full">
                                BEST FOR STUDENTS
                            </div>
                            <div className="mb-6">
                                <h3 className="text-2xl font-bold text-white mb-2">Student Plan</h3>
                                <div className="flex items-baseline mb-2">
                                    <span className="text-5xl font-bold text-cyan-400">₹99</span>
                                    <span className="text-blue-200 ml-2">/month</span>
                                </div>
                                <div className="text-sm text-gray-600 mb-4">
                                    or ₹999/year <span className="text-cyan-400 font-semibold">(Save 17%)</span>
                                </div>
                                <p className="text-gray-600 text-sm">Ideal for students & researchers</p>
                            </div>
                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span><strong>100 jobs/month</strong> (never expire)</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>No daily limit</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>Priority email support</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>Advanced parameters</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>Export to PDF/CSV</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>Activity logs</span>
                                </li>
                            </ul>
                            <Link to="/login" className="btn-cyan block w-full py-3 px-6 text-center rounded-xl shadow-lg">
                                Get Started
                            </Link>
                        </div>

                        {/* Researcher Plan */}
                        <div className="glass-modern p-8 rounded-2xl hover:border-cyan-400/30 transition-all">
                            <div className="mb-6">
                                <div className="text-purple-400 font-bold text-sm mb-2 tracking-wider">POPULAR</div>
                                <h3 className="text-2xl font-bold text-white mb-2">Researcher</h3>
                                <div className="flex items-baseline mb-2">
                                    <span className="text-5xl font-bold text-white">₹499</span>
                                    <span className="text-blue-200 ml-2">/month</span>
                                </div>
                                <div className="text-sm text-gray-600 mb-4">
                                    or ₹4,999/year <span className="text-cyan-400 font-semibold">(Save 17%)</span>
                                </div>
                                <p className="text-gray-600 text-sm">For active researchers & labs</p>
                            </div>
                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span><strong>500 jobs/month</strong> (never expire)</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>No daily limit</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>Priority processing</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>Dedicated support</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>API access</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-cyan-400 mr-2 mt-0.5">✓</span>
                                    <span>Batch uploads & analytics</span>
                                </li>
                            </ul>
                            <Link to="/login" className="btn-cyan-outline block w-full py-3 px-6 text-center rounded-xl">
                                Upgrade Now
                            </Link>
                        </div>

                        {/* Institution Plan */}
                        <div className="glass-modern p-8 rounded-2xl border border-purple-500/30 hover:shadow-2xl transition bg-gradient-to-br from-purple-900/20 to-blue-900/20">
                            <div className="mb-6">
                                <div className="text-yellow-400 font-bold text-sm mb-2 tracking-wider">ENTERPRISE</div>
                                <h3 className="text-2xl font-bold text-white mb-2">Institution</h3>
                                <div className="flex items-baseline mb-2">
                                    <span className="text-5xl font-bold text-white">₹3,999</span>
                                    <span className="text-blue-200 ml-2">/month</span>
                                </div>
                                <div className="text-sm text-gray-600 mb-4">
                                    or ₹39,999/year <span className="text-cyan-400 font-semibold">(Save 17%)</span>
                                </div>
                                <p className="text-gray-600 text-sm">For universities & institutions</p>
                            </div>
                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span><strong>5,000 shared credits</strong></span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span>Multi-user access</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span>Dedicated support channel</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span>Training sessions & SLA</span>
                                </li>
                                <li className="flex items-start text-sm text-gray-700">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span>Custom integrations</span>
                                </li>
                            </ul>
                            <a href="mailto:biodockify@hotmail.com" className="btn-cyan-outline block w-full py-3 px-6 text-center rounded-xl hover:bg-white/10">
                                Contact Sales
                            </a>
                        </div>
                    </div>

                    {/* Credit Value Info */}
                    <div className="mt-12 text-center">
                        <p className="text-blue-200/60 text-sm">
                            💡 <strong>1 Credit = 1 Docking Job</strong> • All plans include free SDF to PDBQT converter • Cancel anytime
                        </p>
                    </div>
                </div>
            </section>
        </div>
    )
}
