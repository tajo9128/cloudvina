import { Link } from 'react-router-dom'

export default function HomePage() {
    return (
        <div className="bg-blue-mesh min-h-screen">
            {/* Hero Section */}
            <section className="pt-12 pb-20 bg-deep-blue-gradient relative overflow-hidden">
                {/* Animated Background Elements */}
                <div className="absolute inset-0 opacity-20">
                    <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500 rounded-full filter blur-3xl animate-float"></div>
                    <div className="absolute bottom-20 right-10 w-96 h-96 bg-cyan-500 rounded-full filter blur-3xl animate-float" style={{ animationDelay: '2s' }}></div>
                </div>

                <div className="container mx-auto px-4 text-center relative z-10">
                    <div className="inline-block bg-tech-cyan-400/20 border border-tech-cyan-400 text-tech-cyan-100 px-6 py-3 rounded-full text-sm font-bold mb-8 shadow-blue-glow">
                        🎉 Get 130 FREE Credits on Signup!
                    </div>
                    <h1 className="text-5xl md:text-7xl font-extrabold text-white mb-6 leading-tight">
                        Molecular Docking <br />
                        <span className="text-blue-gradient">for Indian Researchers</span>
                    </h1>
                    <p className="text-xl text-blue-100 mb-4 max-w-2xl mx-auto">
                        Start with <strong className="text-tech-cyan-400">100 bonus credits</strong> + <strong className="text-tech-cyan-400">30 monthly credits</strong> for free.
                        No credit card required.
                    </p>
                    <p className="text-lg text-blue-200/80 mb-10 max-w-2xl mx-auto">
                        High-performance AutoDock Vina platform powered by AWS. Affordable pricing in ₹. Built for students, researchers, and institutions.
                    </p>
                    <div className="flex flex-wrap justify-center gap-4">
                        <Link to="/login" className="btn-blue-glow text-lg px-10 py-4 rounded-xl font-bold">
                            Start Free with 130 Credits →
                        </Link>
                        <a href="#pricing" className="glass-card text-white px-10 py-4 rounded-xl text-lg font-bold border-blue-glow hover:bg-white/20 transition">
                            View Pricing
                        </a>
                    </div>
                </div>
            </section>

            {/* Free Tools Section */}
            <section className="py-20 bg-gradient-to-b from-deep-navy-900 to-deep-navy-800">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-white mb-4">Free Research Tools</h2>
                        <p className="text-blue-200 max-w-2xl mx-auto">
                            Essential utilities for computational chemistry, available for free.
                        </p>
                    </div>
                    <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8">
                        <div className="glass-card-light card-3d p-8">
                            <div className="text-5xl mb-4">🧪</div>
                            <h3 className="text-2xl font-bold text-deep-navy-900 mb-3">SDF to PDBQT Converter</h3>
                            <p className="text-gray-700 mb-6">
                                Convert your ligand files from SDF format to PDBQT format instantly.
                                Optimized for AutoDock Vina compatibility.
                            </p>
                            <Link to="/tools/converter" className="text-blue-600 font-bold hover:text-blue-700 flex items-center group">
                                Try Converter <span className="ml-2 group-hover:translate-x-1 transition-transform inline-block">→</span>
                            </Link>
                        </div>
                        <div className="glass-card p-8 opacity-60">
                            <div className="text-5xl mb-4">📊</div>
                            <h3 className="text-2xl font-bold text-white mb-3">Results Analyzer</h3>
                            <p className="text-blue-200 mb-6">
                                Visualize and analyze your docking results with advanced 3D rendering.
                                (Coming Soon)
                            </p>
                            <span className="text-blue-300 font-bold cursor-not-allowed">
                                Coming Soon
                            </span>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section id="features" className="py-20 bg-gradient-to-b from-deep-navy-800 to-royal-blue-700">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-white mb-4">Why BioDockify?</h2>
                        <p className="text-blue-100 max-w-2xl mx-auto">
                            Built for researchers who needspeed, reliability, and flexibility.
                        </p>
                    </div>
                    <div className="grid md:grid-cols-3 gap-8">
                        <div className="glass-card-light card-3d p-8">
                            <div className="text-5xl mb-4">🎁</div>
                            <h3 className="text-2xl font-bold text-deep-navy-900 mb-3">Generous Free Tier</h3>
                            <p className="text-gray-700">
                                <strong className="text-blue-600">100 bonus credits</strong> on signup (30-day expiry) + <strong className="text-blue-600">30 monthly credits</strong> (recurring).
                                Start with 130 credits for free!
                            </p>
                        </div>
                        <div className="glass-card-light card-3d p-8">
                            <div className="text-5xl mb-4">⚡</div>
                            <h3 className="text-2xl font-bold text-deep-navy-900 mb-3">Smart Rate Limiting</h3>
                            <p className="text-gray-700">
                                Free users: <strong className="text-blue-600">3 jobs/day</strong> for first month, then 1/day.
                                Paid users: <strong className="text-blue-600">unlimited daily jobs</strong> - use credits anytime!
                            </p>
                        </div>
                        <div className="glass-card-light card-3d p-8">
                            <div className="text-5xl mb-4">💎</div>
                            <h3 className="text-2xl font-bold text-deep-navy-900 mb-3">Credits Never Expire</h3>
                            <p className="text-gray-700">
                                Paid plan credits <strong className="text-blue-600">never expire</strong>. Use them at your own pace.
                                Auto-downgrade to free tier when exhausted.
                            </p>
                        </div>
                        <div className="glass-card-light card-3d p-8">
                            <div className="text-5xl mb-4">🔒</div>
                            <h3 className="text-2xl font-bold text-deep-navy-900 mb-3">Secure & Verified</h3>
                            <p className="text-gray-700">
                                Email verification required for all users.
                                Your data is encrypted and containers destroyed after use.
                            </p>
                        </div>
                        <div className="glass-card-light card-3d p-8">
                            <div className="text-5xl mb-4">🇮🇳</div>
                            <h3 className="text-2xl font-bold text-deep-navy-900 mb-3">Built for India</h3>
                            <p className="text-gray-700">
                                Affordable pricing in <strong className="text-blue-600">Indian Rupees</strong> (₹).
                                Student plans starting at just ₹99/month. Special rates for institutions.
                            </p>
                        </div>
                        <div className="glass-card-light card-3d p-8">
                            <div className="text-5xl mb-4">⚙️</div>
                            <h3 className="text-2xl font-bold text-deep-navy-900 mb-3">AWS Powered</h3>
                            <p className="text-gray-700">
                                Powered by AWS Fargate. Run hundreds of docking jobs in parallel without waiting for queues.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Pricing Section */}
            <section id="pricing" className="py-20 bg-royal-gradient relative overflow-hidden">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-5xl font-bold text-white mb-4">Simple, Transparent Pricing</h2>
                        <p className="text-xl text-blue-100">Start free, scale as you grow. Perfect for Indian students and researchers.</p>
                    </div>
                    <div className="max-w-7xl mx-auto grid md:grid-cols-4 gap-6">
                        {/* Free Tier */}
                        <div className="bg-white p-8 rounded-2xl border-2 border-gray-200 hover:border-purple-300 hover:shadow-lg transition">
                            <div className="mb-6">
                                <div className="text-green-600 font-bold text-sm mb-2">FREE FOREVER</div>
                                <h3 className="text-2xl font-bold text-gray-900 mb-2">Free Plan</h3>
                                <div className="flex items-baseline mb-4">
                                    <span className="text-5xl font-bold text-gray-900">₹0</span>
                                </div>
                                <p className="text-gray-600 text-sm">Perfect for students getting started</p>
                            </div>
                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start text-sm">
                                    <span className="text-green-600 mr-2 mt-0.5">✓</span>
                                    <span><strong>100 bonus credits</strong> (30-day signup bonus)</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-green-600 mr-2 mt-0.5">✓</span>
                                    <span><strong>30 credits/month</strong> (recurring, 1/day)</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-green-600 mr-2 mt-0.5">✓</span>
                                    <span><strong>3 jobs/day</strong> (first month)</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-green-600 mr-2 mt-0.5">✓</span>
                                    <span>1 job/day (after 1st month)</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-green-600 mr-2 mt-0.5">✓</span>
                                    <span>Email verification</span>
                                </li>
                            </ul>
                            <Link to="/login" className="block w-full py-3 px-6 text-center border-2 border-purple-600 text-purple-600 font-bold rounded-xl hover:bg-purple-50 transition">
                                Start Free
                            </Link>
                        </div>

                        {/* Student Plan */}
                        <div className="bg-white p-8 rounded-2xl border-2 border-purple-500 hover:shadow-xl transition relative">
                            <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-purple-600 text-white text-xs font-bold px-4 py-1 rounded-full">
                                BEST FOR STUDENTS
                            </div>
                            <div className="mb-6">
                                <h3 className="text-2xl font-bold text-gray-900 mb-2">Student Plan</h3>
                                <div className="flex items-baseline mb-2">
                                    <span className="text-5xl font-bold text-purple-600">₹99</span>
                                    <span className="text-gray-500 ml-2">/month</span>
                                </div>
                                <div className="text-sm text-gray-600 mb-4">
                                    or ₹999/year <span className="text-green-600 font-semibold">(Save 17%)</span>
                                </div>
                                <p className="text-gray-600 text-sm">Ideal for students & researchers</p>
                            </div>
                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start text-sm">
                                    <span className="text-purple-600 mr-2 mt-0.5">✓</span>
                                    <span><strong>100 jobs/month</strong> (never expire)</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-purple-600 mr-2 mt-0.5">✓</span>
                                    <span>No daily limit</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-purple-600 mr-2 mt-0.5">✓</span>
                                    <span>Priority email support</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-purple-600 mr-2 mt-0.5">✓</span>
                                    <span>Advanced parameters</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-purple-600 mr-2 mt-0.5">✓</span>
                                    <span>Export to PDF/CSV</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-purple-600 mr-2 mt-0.5">✓</span>
                                    <span>Activity logs</span>
                                </li>
                            </ul>
                            <Link to="/login" className="block w-full py-3 px-6 text-center bg-purple-600 text-white font-bold rounded-xl hover:bg-purple-700 transition shadow-lg">
                                Get Started
                            </Link>
                        </div>

                        {/* Researcher Plan */}
                        <div className="bg-white p-8 rounded-2xl border-2 border-gray-200 hover:border-purple-300 hover:shadow-lg transition">
                            <div className="mb-6">
                                <div className="text-blue-600 font-bold text-sm mb-2">POPULAR</div>
                                <h3 className="text-2xl font-bold text-gray-900 mb-2">Researcher</h3>
                                <div className="flex items-baseline mb-2">
                                    <span className="text-5xl font-bold text-gray-900">₹499</span>
                                    <span className="text-gray-500 ml-2">/month</span>
                                </div>
                                <div className="text-sm text-gray-600 mb-4">
                                    or ₹4,999/year <span className="text-green-600 font-semibold">(Save 17%)</span>
                                </div>
                                <p className="text-gray-600 text-sm">For active researchers & labs</p>
                            </div>
                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start text-sm">
                                    <span className="text-blue-600 mr-2 mt-0.5">✓</span>
                                    <span><strong>500 jobs/month</strong> (never expire)</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-blue-600 mr-2 mt-0.5">✓</span>
                                    <span>No daily limit</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-blue-600 mr-2 mt-0.5">✓</span>
                                    <span>Priority processing</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-blue-600 mr-2 mt-0.5">✓</span>
                                    <span>Dedicated support</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-blue-600 mr-2 mt-0.5">✓</span>
                                    <span>API access</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-blue-600 mr-2 mt-0.5">✓</span>
                                    <span>Batch uploads & analytics</span>
                                </li>
                            </ul>
                            <Link to="/login" className="block w-full py-3 px-6 text-center bg-gray-900 text-white font-bold rounded-xl hover:bg-gray-800 transition">
                                Upgrade Now
                            </Link>
                        </div>

                        {/* Institution Plan */}
                        <div className="bg-gradient-to-br from-gray-900 to-gray-800 p-8 rounded-2xl border-2 border-gray-700 hover:shadow-2xl transition text-white">
                            <div className="mb-6">
                                <div className="text-yellow-400 font-bold text-sm mb-2">ENTERPRISE</div>
                                <h3 className="text-2xl font-bold mb-2">Institution</h3>
                                <div className="flex items-baseline mb-2">
                                    <span className="text-5xl font-bold">₹3,999</span>
                                    <span className="text-gray-400 ml-2">/month</span>
                                </div>
                                <div className="text-sm text-gray-400 mb-4">
                                    or ₹39,999/year <span className="text-green-400 font-semibold">(Save 17%)</span>
                                </div>
                                <p className="text-gray-300 text-sm">For universities & institutions</p>
                            </div>
                            <ul className="space-y-3 mb-8">
                                <li className="flex items-start text-sm">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span><strong>5,000 shared credits</strong></span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span>Multi-user access</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span>Dedicated support channel</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span>Training sessions & SLA</span>
                                </li>
                                <li className="flex items-start text-sm">
                                    <span className="text-yellow-400 mr-2 mt-0.5">✓</span>
                                    <span>Custom integrations</span>
                                </li>
                            </ul>
                            <a href="mailto:BioDockify2025@gmail.com" className="block w-full py-3 px-6 text-center bg-white text-gray-900 font-bold rounded-xl hover:bg-gray-100 transition">
                                Contact Sales
                            </a>
                        </div>
                    </div>

                    {/* Credit Value Info */}
                    <div className="mt-12 text-center">
                        <p className="text-gray-600 text-sm">
                            💡 <strong>1 Credit = 1 Docking Job</strong> • All plans include free SDF to PDBQT converter • Cancel anytime
                        </p>
                    </div>
                </div>
            </section>
        </div>
    )
}
