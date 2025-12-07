import { Link } from 'react-router-dom'
import { useState, useEffect } from 'react'
import CountUp from '../components/CountUp'
import SEOHelmet from '../components/SEOHelmet'

export default function HomePage() {
    const [isVisible, setIsVisible] = useState(false)

    useEffect(() => {
        setIsVisible(true)
    }, [])

    // Schema.org structured data for SEO
    const schemaMarkup = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "SoftwareApplication",
                "name": "BioDockify",
                "applicationCategory": "ScientificApplication",
                "applicationSubCategory": "Molecular Docking Software",
                "operatingSystem": "Web Browser",
                "offers": {
                    "@type": "AggregateOffer",
                    "lowPrice": "0",
                    "highPrice": "199",
                    "priceCurrency": "USD",
                    "offerCount": "3"
                },
                "aggregateRating": {
                    "@type": "AggregateRating",
                    "ratingValue": "4.8",
                    "ratingCount": "127",
                    "bestRating": "5"
                },
                "featureList": [
                    "AutoDock Vina Integration",
                    "Cloud-based Molecular Docking",
                    "Multi-format File Support",
                    "AI-powered Results Explanation",
                    "Publication-ready PDF Reports",
                    "Visual Grid Box Configuration",
                    "Automatic SMILES to PDBQT Conversion"
                ],
                "screenshot": "https://biodockify.com/assets/images/dashboard-interface.png",
                "softwareVersion": "4.0.0",
                "author": {
                    "@type": "Organization",
                    "name": "BioDockify",
                    "url": "https://biodockify.com"
                },
                "provider": {
                    "@type": "Organization",
                    "name": "BioDockify",
                    "url": "https://biodockify.com",
                    "logo": "https://biodockify.com/logo.png"
                }
            },
            {
                "@type": "FAQPage",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": "Is BioDockify really free for students?",
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": "Yes! BioDockify offers 130 free credits monthly for all users, which is enough for dozens of molecular docking simulations. Students can use AutoDock Vina online completely free with no credit card required."
                        }
                    },
                    {
                        "@type": "Question",
                        "name": "What file formats does BioDockify support?",
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": "BioDockify supports PDBQT, PDB, SDF, MOL2, and SMILES formats with automatic conversion. Upload any format and we'll convert it to PDBQT for AutoDock Vina docking."
                        }
                    },
                    {
                        "@type": "Question",
                        "name": "How accurate is online molecular docking compared to local installations?",
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": "BioDockify uses official AutoDock Vina 1.2.5, providing identical accuracy to local installations. Results are publication-ready and suitable for thesis work, journal submissions, and academic research."
                        }
                    },
                    {
                        "@type": "Question",
                        "name": "Can I use BioDockify for my M.Pharm or PhD research?",
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": "Absolutely! BioDockify is designed for academic research. You get AI-powered explanations, publication-ready PDF reports with proper citations, and all the features needed for thesis work and journal publications."
                        }
                    }
                ]
            }
        ]
    }

    return (
        <div className="overflow-hidden bg-slate-50">
            <SEOHelmet
                title="Free Molecular Docking Online | AutoDock Vina Cloud Platform - BioDockify"
                description="Run AutoDock Vina molecular docking online free. Cloud-based drug discovery platform for M.Pharm & PhD students. No installation, 130 free credits monthly. AI-powered results explanation."
                keywords="molecular docking online free, autodock vina online, free molecular docking, protein ligand docking online, drug discovery software online, molecular docking for students, cloud based molecular docking"
                canonical="https://biodockify.com/"
                schema={schemaMarkup}
            />

            {/* Hero Section */}
            <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
                {/* Background Overlay */}
                <div className="absolute inset-0 z-0">
                    <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 opacity-95"></div>
                    <img src="/assets/images/hero-molecular.png" alt="Background" className="w-full h-full object-cover opacity-20 mix-blend-overlay" />
                    <h1 className="text-5xl lg:text-7xl font-bold tracking-tight text-white leading-tight mb-6">
                        Molecular Docking <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-secondary-400">Reimagined for the Cloud</span>
                    </h1>

                    <p className="text-xl text-slate-300 leading-relaxed max-w-2xl mx-auto mb-10">
                        Accelerate your drug discovery pipeline with BioDockify. Run thousands of AutoDock Vina simulations in parallel, scaled instantly on AWS infrastructure.
                    </p>

                    <Link to="/dock/new" className="btn-primary text-lg px-8 py-4 shadow-lg shadow-primary-600/20 hover:shadow-primary-600/40 transition-all">
                        Start Free Simulation
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

            </section >

            {/* Video Modal */}


            {/* Stats Section */}
            <section className="py-10 bg-white border-b border-slate-200 relative z-20 -mt-8 mx-4 lg:mx-auto max-w-6xl rounded-2xl shadow-xl">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-8 divide-x divide-slate-100">
                    {[
                        { label: "Jobs Completed", value: 150, suffix: "+" },
                        { label: "Active Researchers", value: 25, suffix: "+" },
                        { label: "Molecules Docked", value: 2500, suffix: "+" },
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

            {/* SEO Content Section - Student Pain Points & Benefits */}
            <section className="py-20 bg-gradient-to-b from-white to-slate-50">
                <div className="container mx-auto px-4">
                    <div className="max-w-7xl mx-auto">
                        <div className="text-center mb-12">
                            <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">
                                Free Molecular Docking Online For Students & Researchers
                            </h2>
                            <p className="text-xl text-slate-600">
                                AutoDock Vina Made Simple, Powerful, and Completely Free
                            </p>
                        </div>

                        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                            {/* Card 1 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 hover:border-primary-200 transition-all group hover:shadow-md">
                                <div className="w-14 h-14 bg-red-100 text-red-600 rounded-2xl flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition-transform">
                                    🚀
                                </div>
                                <h3 className="text-xl font-bold text-slate-900 mb-3">Instant Access, Zero Setup</h3>
                                <p className="text-slate-600 leading-relaxed">
                                    No more complex installations or Linux command lines. Run AutoDock Vina directly from your browser on any device.
                                </p>
                            </div>

                            {/* Card 2 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 hover:border-primary-200 transition-all group hover:shadow-md">
                                <div className="w-14 h-14 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition-transform">
                                    ☁️
                                </div>
                                <h3 className="text-xl font-bold text-slate-900 mb-3">Cloud-Powered Speed</h3>
                                <p className="text-slate-600 leading-relaxed">
                                    Run simulations on AWS infrastructure. What takes hours on a laptop finishes in minutes on our dedicated cloud servers.
                                </p>
                            </div>

                            {/* Card 3 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 hover:border-primary-200 transition-all group hover:shadow-md">
                                <div className="w-14 h-14 bg-green-100 text-green-600 rounded-2xl flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition-transform">
                                    🎓
                                </div>
                                <h3 className="text-xl font-bold text-slate-900 mb-3">Built for Learning</h3>
                                <p className="text-slate-600 leading-relaxed">
                                    AI-powered explanations help you understand binding affinities and interactions. Perfect for thesis work and presentations.
                                </p>
                            </div>

                            {/* Card 4 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 hover:border-primary-200 transition-all group hover:shadow-md">
                                <div className="w-14 h-14 bg-purple-100 text-purple-600 rounded-2xl flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition-transform">
                                    💎
                                </div>
                                <h3 className="text-xl font-bold text-slate-900 mb-3">Unlimited Research</h3>
                                <p className="text-slate-600 leading-relaxed">
                                    Upgrade for high-throughput screening. Run thousands of compounds in parallel with priority queue access.
                                </p>
                            </div>
                        </div>

                    </div>
                </div>
            </section>

            {/* Comprehensive Pipeline Section */}
            <section id="pipeline" className="py-24 bg-white relative overflow-hidden">
                <div className="container mx-auto px-4">
                    <div className="text-center max-w-4xl mx-auto mb-20">
                        <h2 className="text-primary-600 font-bold tracking-wide uppercase text-sm mb-3">End-to-End Workflow</h2>
                        <h3 className="text-3xl md:text-5xl font-bold text-slate-900 mb-6">The Zero-Cost Drug Discovery Pipeline</h3>
                        <p className="text-xl text-slate-600 leading-relaxed">
                            A fully integrated, cloud-native platform that democratizes access to industrial-grade cheminformatics tools. From screening to reporting, we automate the complex biology so you can focus on the discovery.
                        </p>
                    </div>

                    <div className="relative">
                        {/* Connecting Line (Mobile hidden) */}
                        <div className="hidden lg:block absolute left-1/2 transform -translate-x-1/2 h-full w-1 bg-gradient-to-b from-primary-100 via-primary-500 to-primary-100 rounded-full"></div>

                        <div className="space-y-24">
                            {/* Phase 1 */}
                            <div className="relative flex flex-col lg:flex-row items-center justify-between gap-12 group">
                                <div className="lg:w-5/12 text-right order-2 lg:order-1">
                                    <div className="inline-block px-4 py-1.5 rounded-full bg-blue-100 text-blue-800 text-sm font-bold mb-4">Phase 1</div>
                                    <h4 className="text-2xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">High-Throughput Virtual Screening (HTVS)</h4>
                                    <p className="text-slate-600 leading-relaxed">
                                        Parallelized molecular docking using <strong>AutoDock Vina 1.2.5</strong>. Our architecture scales instantly on AWS Batch to screen multi-ligand libraries against your target receptor with microsecond latency.
                                    </p>
                                </div>
                                <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-12 h-12 rounded-full bg-white border-4 border-primary-500 shadow-xl z-10 order-1 lg:order-2">
                                    <span className="text-xl">🧬</span>
                                </div>
                                <div className="lg:w-5/12 order-3 bg-slate-50 rounded-2xl p-6 border border-slate-100 shadow-sm group-hover:shadow-md transition-shadow">
                                    <h5 className="font-bold text-slate-700 mb-2 border-b pb-2">Key Capabilities</h5>
                                    <ul className="space-y-2 text-sm text-slate-600">
                                        <li className="flex items-center gap-2">✓ Automated PDBQT Preparation</li>
                                        <li className="flex items-center gap-2">✓ Custom Grid Box Configuration</li>
                                        <li className="flex items-center gap-2">✓ Multi-Ligand Batch Processing</li>
                                    </ul>
                                </div>
                            </div>

                            {/* Phase 2 */}
                            <div className="relative flex flex-col lg:flex-row items-center justify-between gap-12 group">
                                <div className="lg:w-5/12 order-3 lg:order-1 bg-slate-50 rounded-2xl p-6 border border-slate-100 shadow-sm group-hover:shadow-md transition-shadow">
                                    <h5 className="font-bold text-slate-700 mb-2 border-b pb-2">Key Capabilities</h5>
                                    <ul className="space-y-2 text-sm text-slate-600">
                                        <li className="flex items-center gap-2">✓ Automated Topology Generation (Amber14)</li>
                                        <li className="flex items-center gap-2">✓ Energy Minimization & Equilibration</li>
                                        <li className="flex items-center gap-2">✓ Production Runs (1ns - 100ns)</li>
                                    </ul>
                                </div>
                                <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-12 h-12 rounded-full bg-white border-4 border-primary-500 shadow-xl z-10 order-1 lg:order-2">
                                    <span className="text-xl">⚡</span>
                                </div>
                                <div className="lg:w-5/12 text-left order-2 lg:order-3">
                                    <div className="inline-block px-4 py-1.5 rounded-full bg-purple-100 text-purple-800 text-sm font-bold mb-4">Phase 2</div>
                                    <h4 className="text-2xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">Molecular Dynamics Simulation</h4>
                                    <p className="text-slate-600 leading-relaxed">
                                        Full-atomistic simulations powered by <strong>OpenMM</strong>. Assess the temporal stability of ligand-protein complexes in explicit solvent to filter out false positives from docking.
                                    </p>
                                </div>
                            </div>

                            {/* Phase 3 */}
                            <div className="relative flex flex-col lg:flex-row items-center justify-between gap-12 group">
                                <div className="lg:w-5/12 text-right order-2 lg:order-1">
                                    <div className="inline-block px-4 py-1.5 rounded-full bg-indigo-100 text-indigo-800 text-sm font-bold mb-4">Phase 3</div>
                                    <h4 className="text-2xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">Trajectory Interactivome Analysis</h4>
                                    <p className="text-slate-600 leading-relaxed">
                                        Deep dive into structural dynamics using <strong>MDAnalysis</strong> and PLIP. We quantify RMSD stability, RMSF flexibility, and track hydrogen bond lifetimes to validate binding modes.
                                    </p>
                                </div>
                                <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-12 h-12 rounded-full bg-white border-4 border-primary-500 shadow-xl z-10 order-1 lg:order-2">
                                    <span className="text-xl">📊</span>
                                </div>
                                <div className="lg:w-5/12 order-3 bg-slate-50 rounded-2xl p-6 border border-slate-100 shadow-sm group-hover:shadow-md transition-shadow">
                                    <h5 className="font-bold text-slate-700 mb-2 border-b pb-2">Key Capabilities</h5>
                                    <ul className="space-y-2 text-sm text-slate-600">
                                        <li className="flex items-center gap-2">✓ RMSD & RMSF Plotting</li>
                                        <li className="flex items-center gap-2">✓ H-Bond & Hydrophobic Contact Tracking</li>
                                        <li className="flex items-center gap-2">✓ 3D Trajectory Visualization (DCD)</li>
                                    </ul>
                                </div>
                            </div>

                            {/* Phase 4 */}
                            <div className="relative flex flex-col lg:flex-row items-center justify-between gap-12 group">
                                <div className="lg:w-5/12 order-3 lg:order-1 bg-slate-50 rounded-2xl p-6 border border-slate-100 shadow-sm group-hover:shadow-md transition-shadow">
                                    <h5 className="font-bold text-slate-700 mb-2 border-b pb-2">Key Capabilities</h5>
                                    <ul className="space-y-2 text-sm text-slate-600">
                                        <li className="flex items-center gap-2">✓ Implicit Solvent Models (GB/SA)</li>
                                        <li className="flex items-center gap-2">✓ Enthalpic & Entropic Contributions</li>
                                        <li className="flex items-center gap-2">✓ Improved Ranking Accuracy</li>
                                    </ul>
                                </div>
                                <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-12 h-12 rounded-full bg-white border-4 border-primary-500 shadow-xl z-10 order-1 lg:order-2">
                                    <span className="text-xl">🔥</span>
                                </div>
                                <div className="lg:w-5/12 text-left order-2 lg:order-3">
                                    <div className="inline-block px-4 py-1.5 rounded-full bg-orange-100 text-orange-800 text-sm font-bold mb-4">Phase 4</div>
                                    <h4 className="text-2xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">MM-GBSA Binding Free Energy</h4>
                                    <p className="text-slate-600 leading-relaxed">
                                        Move beyond simple scoring functions. We calculate rigorous Binding Free Energy (ΔG) from MD trajectories to provide a thermodynamically accurate ranking of your top hits.
                                    </p>
                                </div>
                            </div>

                            {/* Phase 5 */}
                            <div className="relative flex flex-col lg:flex-row items-center justify-between gap-12 group">
                                <div className="lg:w-5/12 text-right order-2 lg:order-1">
                                    <div className="inline-block px-4 py-1.5 rounded-full bg-teal-100 text-teal-800 text-sm font-bold mb-4">Phase 5</div>
                                    <h4 className="text-2xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">Consensus Lead Ranking</h4>
                                    <p className="text-slate-600 leading-relaxed">
                                        Our <strong>Ranking Engine</strong> aggregates data from Docking, MD, and MM-GBSA to compute a weighted consensus score. This data-driven approach minimizes false positives and highlights true potential.
                                    </p>
                                </div>
                                <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-12 h-12 rounded-full bg-white border-4 border-primary-500 shadow-xl z-10 order-1 lg:order-2">
                                    <span className="text-xl">🏆</span>
                                </div>
                                <div className="lg:w-5/12 order-3 bg-slate-50 rounded-2xl p-6 border border-slate-100 shadow-sm group-hover:shadow-md transition-shadow">
                                    <h5 className="font-bold text-slate-700 mb-2 border-b pb-2">Key Capabilities</h5>
                                    <ul className="space-y-2 text-sm text-slate-600">
                                        <li className="flex items-center gap-2">✓ Weighted Scoring Algorithms</li>
                                        <li className="flex items-center gap-2">✓ Normalization & Standardization</li>
                                        <li className="flex items-center gap-2">✓ Interactive Leaderboard</li>
                                    </ul>
                                </div>
                            </div>

                            {/* Phase 6 */}
                            <div className="relative flex flex-col lg:flex-row items-center justify-between gap-12 group">
                                <div className="lg:w-5/12 order-3 lg:order-1 bg-slate-50 rounded-2xl p-6 border border-slate-100 shadow-sm group-hover:shadow-md transition-shadow">
                                    <h5 className="font-bold text-slate-700 mb-2 border-b pb-2">Key Capabilities</h5>
                                    <ul className="space-y-2 text-sm text-slate-600">
                                        <li className="flex items-center gap-2">✓ Blood-Brain Barrier (BBB) Permeability</li>
                                        <li className="flex items-center gap-2">✓ Structural Toxicity Alerts (PAINS)</li>
                                        <li className="flex items-center gap-2">✓ Drug-Likeness (QED) Profiling</li>
                                    </ul>
                                </div>
                                <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-12 h-12 rounded-full bg-white border-4 border-primary-500 shadow-xl z-10 order-1 lg:order-2">
                                    <span className="text-xl">🛡️</span>
                                </div>
                                <div className="lg:w-5/12 text-left order-2 lg:order-3">
                                    <div className="inline-block px-4 py-1.5 rounded-full bg-red-100 text-red-800 text-sm font-bold mb-4">Phase 6</div>
                                    <h4 className="text-2xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">ADMET & Safety Profiling</h4>
                                    <p className="text-slate-600 leading-relaxed">
                                        Fail early, fail cheap. We integrate <strong>RDKit</strong> and machine learning models to predict pharmacokinetic properties and toxicity risks before you enter the wet lab.
                                    </p>
                                </div>
                            </div>

                            {/* Phase 7 */}
                            <div className="relative flex flex-col lg:flex-row items-center justify-between gap-12 group">
                                <div className="lg:w-5/12 text-right order-2 lg:order-1">
                                    <div className="inline-block px-4 py-1.5 rounded-full bg-green-100 text-green-800 text-sm font-bold mb-4">Phase 7</div>
                                    <h4 className="text-2xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">Automated Consensus Reporting</h4>
                                    <p className="text-slate-600 leading-relaxed">
                                        Generate publication-ready PDF reports with a single click. Summarize your top candidates, including structure visualizations, affinity scores, and ADMET profiles, ready for hand-off.
                                    </p>
                                </div>
                                <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-12 h-12 rounded-full bg-white border-4 border-primary-500 shadow-xl z-10 order-1 lg:order-2">
                                    <span className="text-xl">📑</span>
                                </div>
                                <div className="lg:w-5/12 order-3 bg-slate-50 rounded-2xl p-6 border border-slate-100 shadow-sm group-hover:shadow-md transition-shadow">
                                    <h5 className="font-bold text-slate-700 mb-2 border-b pb-2">Key Capabilities</h5>
                                    <ul className="space-y-2 text-sm text-slate-600">
                                        <li className="flex items-center gap-2">✓ One-Click PDF Generation</li>
                                        <li className="flex items-center gap-2">✓ 2D Structure Rendering</li>
                                        <li className="flex items-center gap-2">✓ Comprehensive Data Summary</li>
                                    </ul>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>
            </section>
            <section className="py-24 bg-slate-50 border-t border-slate-200">
                <div className="container mx-auto px-4">
                    <div className="grid lg:grid-cols-2 gap-16 items-start">
                        {/* Left Column: Testimonials */}
                        <div>
                            <div className="mb-10">
                                <h2 className="text-primary-600 font-bold tracking-wide uppercase text-sm mb-3">Community Love</h2>
                                <h3 className="text-3xl font-bold text-slate-900">Trusted by Researchers</h3>
                            </div>

                            {/* Testimonial 1 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 italic text-slate-600 relative">
                                <span className="absolute top-4 left-4 text-6xl text-primary-100 font-serif leading-none opacity-50">"</span>
                                <p className="mb-4 relative z-10">
                                    "The **Zero-Cost Pipeline** is a game changer. I used Phase 2 (MD Simulation) to validate my docking results for a high-impact factor publication. It saved me months of work."
                                </p>
                                <div className="flex items-center gap-3 not-italic">
                                    <div className="w-10 h-10 rounded-full bg-slate-200 flex items-center justify-center font-bold text-slate-500">
                                        AP
                                    </div>
                                    <div>
                                        <div className="font-bold text-slate-900 text-sm">Amit Patel</div>
                                        <div className="text-xs text-slate-500">M.Pharm Student, NIPER</div>
                                    </div>
                                </div>
                            </div>

                            {/* Testimonial 2 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 italic text-slate-600 relative">
                                <span className="absolute top-4 left-4 text-6xl text-secondary-100 font-serif leading-none opacity-50">"</span>
                                <p className="mb-4 relative z-10">
                                    "Phase 6 (ADMET Profiling) flagged a critical toxicity issue in my top lead before we ordered synthesis. This platform is an essential tool for **de-risking drug discovery**."
                                </p>
                                <div className="flex items-center gap-3 not-italic">
                                    <div className="w-10 h-10 rounded-full bg-slate-200 flex items-center justify-center font-bold text-slate-500">
                                        SR
                                    </div>
                                    <div>
                                        <div className="font-bold text-slate-900 text-sm">Dr. Sneha Rao</div>
                                        <div className="text-xs text-slate-500">Research Scientist, Aurigene</div>
                                    </div>
                                </div>
                            </div>

                            {/* Testimonial 3 */}
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 italic text-slate-600 relative">
                                <span className="absolute top-4 left-4 text-6xl text-teal-100 font-serif leading-none opacity-50">"</span>
                                <p className="mb-4 relative z-10">
                                    "The **Consensus Reporting** (Phase 7) generated a PDF that I attached directly to my thesis. The RDKit structure renderings look professional and publication-ready."
                                </p>
                                <div className="flex items-center gap-3 not-italic">
                                    <div className="w-10 h-10 rounded-full bg-slate-200 flex items-center justify-center font-bold text-slate-500">
                                        RK
                                    </div>
                                    <div>
                                        <div className="font-bold text-slate-900 text-sm">Rajesh Kumar</div>
                                        <div className="text-xs text-slate-500">PhD Scholar, IIT Delhi</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Right Column: FAQs */}
                        <div>
                            <div className="mb-10">
                                <h2 className="text-primary-600 font-bold tracking-wide uppercase text-sm mb-3">Common Questions</h2>
                                <h3 className="text-3xl font-bold text-slate-900">Frequently Asked Questions</h3>
                            </div>

                            <div className="space-y-4">
                                {[
                                    {
                                        q: "Is the entire 7-Phase Pipeline really free?",
                                        a: "Yes. Our 'Zero-Cost' mission means Phases 1 through 7 (Docking, MD, Analysis, Ranking, ADMET, Reporting) are available on the free tier, with generous monthly credits."
                                    },
                                    {
                                        q: "How accurate is the MM-GBSA Binding Energy (Phase 4)?",
                                        a: "We use OpenMM's implicit solvent models to calculate ΔG. While not as rigorous as FEP, it provides a significantly better ranking metric than simple docking scores."
                                    },
                                    {
                                        q: "Can I download the raw simulation data?",
                                        a: "Absolutely. You own your data. Download PDBQT files, DCD trajectories, and PDF reports at any time for offline analysis."
                                    },
                                    {
                                        q: "Is my data secure on the cloud?",
                                        a: "We use AWS S3 with strict encryption. Your molecular structures and results are private and deleted automatically after 30 days unless saved."
                                    }
                                ].map((faq, i) => (
                                    <div key={i} className="bg-white rounded-xl p-6 shadow-sm border border-slate-200 hover:border-primary-200 transition-all">
                                        <h4 className="font-bold text-slate-900 mb-2 flex items-start gap-3 text-sm md:text-base">
                                            <span className="text-primary-500 text-lg leading-none">Q.</span>
                                            {faq.q}
                                        </h4>
                                        <p className="text-slate-600 text-sm leading-relaxed pl-7">
                                            {faq.a}
                                        </p>
                                    </div>
                                ))}
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
                            <h4 className="text-3xl font-bold text-white mb-2">₹0 <span className="text-lg text-slate-400 font-normal">/ month</span></h4>
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
                            <h4 className="text-3xl font-bold text-white mb-2">₹49 <span className="text-lg text-slate-400 font-normal">/ month</span></h4>
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
                            <h4 className="text-3xl font-bold text-white mb-2">₹199 <span className="text-lg text-slate-400 font-normal">/ month</span></h4>
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

        </div >
    )
}
