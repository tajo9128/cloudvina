import { Link } from 'react-router-dom'

import { Link } from 'react-router-dom'

export default function HomePage() {
    return (
        <div className="bg-white">
            {/* Hero Section */}
            <section className="pt-12 pb-20 bg-gradient-to-b from-purple-50 to-white">
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
            <section id="pricing" className="py-20 bg-gradient-to-b from-white to-purple-50">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-gray-900 mb-4">Simple, Transparent Pricing</h2>
                        <p className="text-xl text-gray-600">Start free, scale as you grow. Perfect for Indian students and researchers.</p>
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
                                    <span>Email & phone verification</span>
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
                            <a href="mailto:cloudvina2025@gmail.com" className="block w-full py-3 px-6 text-center bg-white text-gray-900 font-bold rounded-xl hover:bg-gray-100 transition">
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
