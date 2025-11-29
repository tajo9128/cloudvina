import { Link } from 'react-router-dom'
import { useState, useEffect } from 'react'
import CountUp from '../components/CountUp'

export default function HomePage() {
    const [isVisible, setIsVisible] = useState(false)

    useEffect(() => {
        setIsVisible(true)
    }, [])

    return (
        <div className="overflow-hidden bg-slate-50">
            {/* Hero Section */}
            <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
                {/* Background Overlay */}
                <div className="absolute inset-0 z-0">
                    <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 opacity-95"></div>
                    <img
                        src="/assets/images/hero-molecular.png"
                        alt="Background"
                        className="w-full h-full object-cover opacity-20 mix-blend-overlay"
                    />
                    <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
                </div>

                <div className="container mx-auto px-4 relative z-10 text-center">
                    <div className={`transition-all duration-1000 transform ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-500/10 border border-primary-500/20 text-primary-300 font-medium text-sm mb-8 backdrop-blur-sm">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary-500"></span>
                            </span>
                            v2.0 Now Live: Multi-Format Support
                        </div>

                        <h1 className="text-5xl lg:text-7xl font-bold tracking-tight text-white leading-tight mb-6">
                            Molecular Docking <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-secondary-400">Reimagined for the Cloud</span>
                        </h1>

                        <p className="text-xl text-slate-300 leading-relaxed max-w-2xl mx-auto mb-10">
                            Accelerate your drug discovery pipeline with BioDockify. Run thousands of AutoDock Vina simulations in parallel, scaled instantly on AWS infrastructure.
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <Link to="/dock/new" className="btn-primary text-lg px-8 py-4 shadow-lg shadow-primary-600/20 hover:shadow-primary-600/40 transition-all">
                                Start Free Simulation
                            </Link>
                            <Link to="/#features" className="px-8 py-4 rounded-xl font-bold text-white border border-slate-600 hover:bg-slate-800 transition-all">
                                Explore Features
                            </Link>
                        </div>

                        <div className="mt-12 text-slate-400 text-sm font-medium flex justify-center gap-8">
                            <span className="flex items-center gap-2">
                                <svg className="w-5 h-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                No Credit Card Required
                            </span>
                            <span className="flex items-center gap-2">
                                <svg className="w-5 h-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                130 Free Credits
                            </span>
                        </div>
                    </div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="py-10 bg-white border-b border-slate-200 relative z-20 -mt-8 mx-4 lg:mx-auto max-w-6xl rounded-2xl shadow-xl">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-8 divide-x divide-slate-100">
                    {[
                        { label: "Jobs Completed", value: 10000, suffix: "+" },
                        { label: "Active Researchers", value: 500, suffix: "+" },
                        { label: "Molecules Docked", value: 50000, suffix: "+" },
                        { label: "Uptime", value: 99.9, suffix: "%" }
                    ].map((stat, i) => (
                        <div key={i} className="text-center px-4">
                            <div className="text-3xl lg:text-4xl font-bold text-slate-900 mb-1">
                                <CountUp end={stat.value} suffix={stat.suffix} />
                            </div>
                            <div className="text-sm text-slate-500 font-medium uppercase tracking-wide">{stat.label}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Features Section - Alternating Layout */}
            <section id="features" className="py-24 overflow-hidden">
                <div className="container mx-auto px-4">
                    <div className="text-center max-w-3xl mx-auto mb-20">
                        <h2 className="text-primary-600 font-bold tracking-wide uppercase text-sm mb-3">Why BioDockify?</h2>
                        <h3 className="text-3xl md:text-4xl font-bold text-slate-900">Enterprise-grade docking infrastructure</h3>
                    </div>

                    <div className="space-y-24">
                        {/* Feature 1 */}
                        <div className="grid lg:grid-cols-2 gap-16 items-center">
                            <div className="relative">
                                <div className="absolute inset-0 bg-primary-200 rounded-full blur-3xl opacity-20 transform -translate-x-10"></div>
                                <img src="/assets/images/dashboard-interface.png" alt="Cloud Scalability" className="relative rounded-2xl shadow-2xl border border-slate-200" />
                            </div>
                            <div>
                                <div className="w-12 h-12 rounded-xl bg-primary-100 text-primary-600 flex items-center justify-center text-2xl mb-6">
                                    ☁️
                                </div>
                                <h3 className="text-3xl font-bold text-slate-900 mb-4">Infinite Cloud Scalability</h3>
                                <p className="text-lg text-slate-600 leading-relaxed mb-6">
                                    Forget queuing on local clusters. BioDockify leverages AWS Batch to spin up thousands of instances instantly. Run massive virtual screens in hours, not weeks.
                                </p>
                                <ul className="space-y-3">
                                    {[
                                        "Auto-scaling infrastructure",
                                        "Zero maintenance required",
                                        "Pay only for compute used"
                                    ].map((item, i) => (
                                        <li key={i} className="flex items-center gap-3 text-slate-700">
                                            <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>

                        {/* Feature 2 */}
                        <div className="grid lg:grid-cols-2 gap-16 items-center">
                            <div className="order-2 lg:order-1">
                                <div className="w-12 h-12 rounded-xl bg-secondary-100 text-secondary-600 flex items-center justify-center text-2xl mb-6">
                                    🧬
                                </div>
                                <h3 className="text-3xl font-bold text-slate-900 mb-4">Universal Format Support</h3>
                                <p className="text-lg text-slate-600 leading-relaxed mb-6">
                                    We handle the messy work of file conversion. Upload your receptors and ligands in almost any format, and our automated pipelines prepare them for AutoDock Vina.
                                </p>
                                <div className="flex flex-wrap gap-2">
                                    {['PDB', 'PDBQT', 'SDF', 'MOL2', 'CIF', 'XML', 'SMILES'].map((fmt) => (
                                        <span key={fmt} className="px-3 py-1 rounded-lg bg-slate-100 text-slate-600 text-sm font-bold border border-slate-200">
                                            {fmt}
                                        </span>
                                    ))}
                                </div>
                            </div>
                            <div className="order-1 lg:order-2 relative">
                                <div className="absolute inset-0 bg-secondary-200 rounded-full blur-3xl opacity-20 transform translate-x-10"></div>
                                <img src="/assets/images/hero-molecular.png" alt="Format Support" className="relative rounded-2xl shadow-2xl border border-slate-200" />
                            </div>
                        </div>

                        {/* Feature 3 */}
                        <div className="grid lg:grid-cols-2 gap-16 items-center">
                            <div className="relative">
                                <div className="absolute inset-0 bg-teal-200 rounded-full blur-3xl opacity-20 transform -translate-y-10"></div>
                                {/* Placeholder for analysis image - reusing dashboard for now but styled differently */}
                                <div className="relative rounded-2xl shadow-2xl border border-slate-200 bg-slate-900 p-2 overflow-hidden">
                                    <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-transparent to-transparent z-10"></div>
                                    <img src="/assets/images/dashboard-interface.png" alt="Analysis" className="rounded-xl opacity-80" />
                                    <div className="absolute bottom-6 left-6 right-6 z-20">
                                        <div className="bg-white/10 backdrop-blur-md rounded-xl p-4 border border-white/20 text-white">
                                            <div className="flex justify-between items-center mb-2">
                                                <span className="font-bold">Binding Affinity</span>
                                                <span className="text-green-400 font-mono">-9.4 kcal/mol</span>
                                            </div>
                                            <div className="w-full bg-white/20 rounded-full h-1.5">
                                                <div className="bg-green-400 h-1.5 rounded-full" style={{ width: '85%' }}></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div className="w-12 h-12 rounded-xl bg-teal-100 text-teal-600 flex items-center justify-center text-2xl mb-6">
                                    🔬
                                </div>
                                <h3 className="text-3xl font-bold text-slate-900 mb-4">Interactive 3D Analysis</h3>
                                <p className="text-lg text-slate-600 leading-relaxed mb-6">
                                    Visualize your docking results directly in the browser. No need to download massive files or install complex desktop software just to check a pose.
                                </p>
                                <Link to="/signup" className="text-primary-600 font-bold hover:text-primary-700 inline-flex items-center gap-2 group">
                                    Start Analyzing <span className="transform group-hover:translate-x-1 transition-transform">→</span>
                                </Link>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Pricing Section */}
            <section id="pricing" className="py-24 bg-slate-900 text-white relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
                    <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-primary-500/10 rounded-full blur-3xl"></div>
                    <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] bg-secondary-500/10 rounded-full blur-3xl"></div>
                </div>

                <div className="container mx-auto px-4 relative z-10">
                    <div className="text-center max-w-3xl mx-auto mb-16">
                        <h2 className="text-primary-400 font-bold tracking-wide uppercase text-sm mb-3">Flexible Pricing</h2>
                        <h3 className="text-4xl font-bold text-white mb-6">Research-friendly plans</h3>
                        <p className="text-slate-400 text-lg">
                            Whether you're a student or running a large lab, we have a plan that fits your needs.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
                        {/* Free Tier */}
                        <div className="p-8 rounded-2xl border border-slate-700 bg-slate-800/50 hover:border-primary-500/50 transition-all">
                            <div className="mb-4">
                                <span className="px-3 py-1 rounded-full bg-slate-700 text-slate-300 text-xs font-bold uppercase tracking-wide">Starter</span>
                            </div>
                            <h4 className="text-3xl font-bold text-white mb-2">$0 <span className="text-lg text-slate-400 font-normal">/ month</span></h4>
                            <p className="text-slate-400 mb-6">Perfect for students and testing.</p>
                            <ul className="space-y-4 mb-8">
                                <li className="flex items-center gap-3 text-slate-300">
                                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                    3 Jobs per day
                                </li>
                                <li className="flex items-center gap-3 text-slate-300">
                                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                    Standard Priority
                                </li>
                            </ul>
                            <Link to="/signup" className="w-full btn-secondary bg-slate-700 hover:bg-slate-600 border-none text-white">Get Started</Link>
                        </div>

                        {/* Pro Tier */}
                        <div className="p-8 rounded-2xl border-2 border-primary-500 bg-slate-800 relative transform md:-translate-y-4 shadow-2xl shadow-primary-900/50">
                            <div className="absolute top-0 right-0 bg-primary-500 text-white text-xs font-bold px-3 py-1 rounded-bl-xl rounded-tr-xl">POPULAR</div>
                            <div className="mb-4">
                                <span className="px-3 py-1 rounded-full bg-primary-900/50 text-primary-300 text-xs font-bold uppercase tracking-wide">Researcher</span>
                            </div>
                            <h4 className="text-3xl font-bold text-white mb-2">$49 <span className="text-lg text-slate-400 font-normal">/ month</span></h4>
                            <p className="text-slate-400 mb-6">For serious research projects.</p>
                            <ul className="space-y-4 mb-8">
                                <li className="flex items-center gap-3 text-white">
                                    <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                    100 Jobs per day
                                </li>
                                <li className="flex items-center gap-3 text-white">
                                    <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                    High Priority Queue
                                </li>
                                <li className="flex items-center gap-3 text-white">
                                    <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                    Advanced Analytics
                                </li>
                            </ul>
                            <Link to="/signup?plan=pro" className="w-full btn-primary bg-primary-500 hover:bg-primary-400 border-none">Upgrade Now</Link>
                        </div>

                        {/* Enterprise Tier */}
                        <div className="p-8 rounded-2xl border border-slate-700 bg-slate-800/50 hover:border-primary-500/50 transition-all">
                            <div className="mb-4">
                                <span className="px-3 py-1 rounded-full bg-slate-700 text-slate-300 text-xs font-bold uppercase tracking-wide">Lab / Team</span>
                            </div>
                            <h4 className="text-3xl font-bold text-white mb-2">$199 <span className="text-lg text-slate-400 font-normal">/ month</span></h4>
                            <p className="text-slate-400 mb-6">For labs and large teams.</p>
                            <ul className="space-y-4 mb-8">
                                <li className="flex items-center gap-3 text-slate-300">
                                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                    Unlimited Jobs
                                </li>
                                <li className="flex items-center gap-3 text-slate-300">
                                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                    Dedicated Support
                                </li>
                            </ul>
                            <Link to="/contact" className="w-full btn-secondary bg-slate-700 hover:bg-slate-600 border-none text-white">Contact Sales</Link>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-24 bg-primary-600 relative overflow-hidden">
                <div className="absolute inset-0 opacity-20 bg-[url('/assets/images/hero-molecular.png')] bg-cover bg-center mix-blend-overlay"></div>
                <div className="container mx-auto px-4 relative z-10 text-center">
                    <h2 className="text-4xl font-bold text-white mb-6">Ready to accelerate your research?</h2>
                    <p className="text-xl text-primary-100 mb-10 max-w-2xl mx-auto">
                        Join thousands of researchers using BioDockify to discover new therapeutics faster than ever before.
                    </p>
                    <Link to="/signup" className="bg-white text-primary-600 hover:bg-slate-100 font-bold text-lg px-10 py-4 rounded-xl shadow-xl transition-all inline-flex">
                        Create Free Account
                    </Link>
                </div>
            </section>
        </div>
    )
}
