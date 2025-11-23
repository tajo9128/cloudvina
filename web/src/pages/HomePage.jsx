import { Link } from 'react-router-dom'

export default function HomePage() {
    return (
        <div className="min-h-screen bg-white">
            {/* Header */}
            <header className="bg-white shadow-sm fixed w-full z-50">
                <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                    <Link to="/" className="flex items-center space-x-2 text-gray-800 hover:text-purple-600 transition">
                        <div className="text-2xl">🧬</div>
                        <h1 className="text-xl font-bold">CloudVina</h1>
                    </Link>
                    <nav className="hidden md:flex space-x-8">
                        <a href="#features" className="text-gray-600 hover:text-purple-600 font-medium">Features</a>
                        <a href="#pricing" className="text-gray-600 hover:text-purple-600 font-medium">Pricing</a>
                        <Link to="/tools/converter" className="text-gray-600 hover:text-purple-600 font-medium">Tools</Link>
                        <Link to="/dashboard" className="text-gray-600 hover:text-purple-600 font-medium">Dashboard</Link>
                    </nav>
                    <div className="flex gap-4">
                        <Link to="/login" className="px-4 py-2 text-purple-600 font-medium hover:bg-purple-50 rounded-lg transition">Log In</Link>
                        <Link to="/dock/new" className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition shadow-md">Start Docking</Link>
                    </div>
                </div>
            </header>

            {/* Hero Section */}
            <section className="pt-32 pb-20 bg-gradient-to-b from-purple-50 to-white">
                <div className="container mx-auto px-4 text-center">
                    <h1 className="text-5xl md:text-6xl font-extrabold text-gray-900 mb-6 leading-tight">
                        Molecular Docking <br />
                        <span className="text-purple-600">at Cloud Scale</span>
                    </h1>
                    <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto">
                        Accelerate your drug discovery pipeline with our high-performance, secure, and scalable AutoDock Vina platform. No infrastructure to manage.
                    </p>
                    <div className="flex justify-center gap-4">
                        <Link to="/dock/new" className="bg-purple-600 text-white px-8 py-4 rounded-xl text-lg font-bold hover:bg-purple-700 transition shadow-lg hover:shadow-xl transform hover:-translate-y-1">
                            Launch Job
                        </Link>
                        <a href="#features" className="bg-white text-gray-700 px-8 py-4 rounded-xl text-lg font-bold border border-gray-200 hover:border-purple-200 hover:bg-purple-50 transition shadow-sm">
                            Learn More
                        </a>
                    </div>
                </div>
            </section>

            {/* Free Tools Section */}
            <section className="py-20 bg-white">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-gray-900 mb-4">Free Research Tools</h2>
                        <p className="text-gray-600 max-w-2xl mx-auto">
                            Essential utilities for computational chemistry, available for free.
                        </p>
                    </div>
                    <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8">
                        <div className="bg-purple-50 rounded-2xl p-8 border border-purple-100 hover:shadow-lg transition">
                            <div className="text-4xl mb-4">🧪</div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">SDF to PDBQT Converter</h3>
                            <p className="text-gray-600 mb-6">
                                Convert your ligand files from SDF format to PDBQT format instantly.
                                Optimized for AutoDock Vina compatibility.
                            </p>
                            <Link to="/tools/converter" className="text-purple-600 font-bold hover:text-purple-700 flex items-center">
                                Try Converter <span className="ml-2">→</span>
                            </Link>
                        </div>
                        <div className="bg-gray-50 rounded-2xl p-8 border border-gray-100 opacity-75">
                            <div className="text-4xl mb-4">📊</div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">Results Analyzer</h3>
                            <p className="text-gray-600 mb-6">
                                Visualize and analyze your docking results with advanced 3D rendering.
                                (Coming Soon)
                            </p>
                            <span className="text-gray-400 font-bold cursor-not-allowed">
                                Coming Soon
                            </span>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section id="features" className="py-20 bg-gray-50">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-gray-900 mb-4">Why CloudVina?</h2>
                        <p className="text-gray-600 max-w-2xl mx-auto">
                            Built for researchers who need speed, reliability, and simplicity.
                        </p>
                    </div>
                    <div className="grid md:grid-cols-3 gap-8">
                        <div className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition">
                            <div className="text-4xl mb-4">⚡</div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">Lightning Fast</h3>
                            <p className="text-gray-600">Powered by AWS Fargate. Run hundreds of docking jobs in parallel without waiting for queues.</p>
                        </div>
                        <div className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition">
                            <div className="text-4xl mb-4">🔒</div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">Secure & Private</h3>
                            <p className="text-gray-600">Your data is encrypted in transit and at rest. We use ephemeral containers that are destroyed after use.</p>
                        </div>
                        <div className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition">
                            <div className="text-4xl mb-4">💰</div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">Cost Effective</h3>
                            <p className="text-gray-600">Pay only for the compute you use. No monthly fees or hidden infrastructure costs.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Pricing Section */}
            <section id="pricing" className="py-20 bg-white">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-gray-900 mb-4">Simple, Transparent Pricing</h2>
                        <p className="text-gray-600">Start for free, upgrade as you scale.</p>
                    </div>
                    <div className="max-w-5xl mx-auto grid md:grid-cols-3 gap-8 items-center">
                        <div className="bg-white p-8 rounded-2xl border border-gray-200">
                            <h3 className="text-xl font-bold text-gray-900 mb-2">Researcher</h3>
                            <div className="text-4xl font-bold text-gray-900 mb-4">$0<span className="text-lg text-gray-500 font-normal">/mo</span></div>
                            <ul className="space-y-4 mb-8 text-gray-600">
                                <li className="flex items-center">✓ 10 Free Docking Credits</li>
                                <li className="flex items-center">✓ Standard Priority</li>
                                <li className="flex items-center">✓ Public Support</li>
                            </ul>
                            <Link to="/login" className="block w-full py-3 px-6 text-center border-2 border-purple-600 text-purple-600 font-bold rounded-xl hover:bg-purple-50 transition">
                                Get Started
                            </Link>
                        </div>
                        <div className="bg-purple-600 p-8 rounded-2xl shadow-xl transform scale-105 relative">
                            <div className="absolute top-0 right-0 bg-yellow-400 text-xs font-bold px-3 py-1 rounded-bl-lg rounded-tr-lg text-purple-900">POPULAR</div>
                            <h3 className="text-xl font-bold text-white mb-2">Lab Team</h3>
                            <div className="text-4xl font-bold text-white mb-4">$49<span className="text-lg text-purple-200 font-normal">/mo</span></div>
                            <ul className="space-y-4 mb-8 text-purple-100">
                                <li className="flex items-center">✓ 500 Docking Credits</li>
                                <li className="flex items-center">✓ High Priority Queue</li>
                                <li className="flex items-center">✓ Priority Email Support</li>
                                <li className="flex items-center">✓ API Access</li>
                            </ul>
                            <Link to="/login" className="block w-full py-3 px-6 text-center bg-white text-purple-600 font-bold rounded-xl hover:bg-gray-100 transition">
                                Start Free Trial
                            </Link>
                        </div>
                        <div className="bg-white p-8 rounded-2xl border border-gray-200">
                            <h3 className="text-xl font-bold text-gray-900 mb-2">Enterprise</h3>
                            <div className="text-4xl font-bold text-gray-900 mb-4">Custom</div>
                            <ul className="space-y-4 mb-8 text-gray-600">
                                <li className="flex items-center">✓ Unlimited Credits</li>
                                <li className="flex items-center">✓ Dedicated Instance</li>
                                <li className="flex items-center">✓ SLA Guarantee</li>
                                <li className="flex items-center">✓ 24/7 Phone Support</li>
                            </ul>
                            <a href="mailto:sales@cloudvina.in" className="block w-full py-3 px-6 text-center border-2 border-gray-200 text-gray-600 font-bold rounded-xl hover:border-gray-400 hover:text-gray-800 transition">
                                Contact Sales
                            </a>
                        </div>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="bg-gray-900 text-white py-12">
                <div className="container mx-auto px-4">
                    <div className="grid md:grid-cols-4 gap-8 mb-8">
                        <div>
                            <div className="flex items-center space-x-2 mb-4">
                                <div className="text-2xl">🧬</div>
                                <h2 className="text-xl font-bold">CloudVina</h2>
                            </div>
                            <p className="text-gray-400 text-sm">
                                Democratizing drug discovery with cloud-native molecular docking tools.
                            </p>
                        </div>
                        <div>
                            <h3 className="font-bold mb-4">Product</h3>
                            <ul className="space-y-2 text-gray-400 text-sm">
                                <li><a href="#features" className="hover:text-white">Features</a></li>
                                <li><a href="#pricing" className="hover:text-white">Pricing</a></li>
                                <li><Link to="/tools/converter" className="hover:text-white">SDF Converter</Link></li>
                                <li><Link to="/admin" className="hover:text-white">Admin</Link></li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-bold mb-4">Company</h3>
                            <ul className="space-y-2 text-gray-400 text-sm">
                                <li><Link to="/about" className="hover:text-white">About Us</Link></li>
                                <li><Link to="/blog" className="hover:text-white">Blog</Link></li>
                                <li><a href="mailto:cloudvina2025@gmail.com" className="hover:text-white">Contact</a></li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-bold mb-4">Legal</h3>
                            <ul className="space-y-2 text-gray-400 text-sm">
                                <li><Link to="/privacy" className="hover:text-white">Privacy Policy</Link></li>
                                <li><Link to="/terms" className="hover:text-white">Terms of Service</Link></li>
                            </ul>
                        </div>
                    </div>
                    <div className="border-t border-gray-800 pt-8 text-center text-gray-500 text-sm">
                        © {new Date().getFullYear()} CloudVina. All rights reserved.
                    </div>
                </div>
            </footer>
        </div>
    )
}
