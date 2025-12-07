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
                    }
                ]
            }
        ]
    }

    const PipelinePhase = ({ phase, title, description, icon, align = "left", color = "primary" }) => {
        const colors = {
            primary: "bg-blue-100 text-blue-700 border-blue-200",
            purple: "bg-purple-100 text-purple-700 border-purple-200",
            indigo: "bg-indigo-100 text-indigo-700 border-indigo-200",
            emerald: "bg-emerald-100 text-emerald-700 border-emerald-200",
            orange: "bg-orange-100 text-orange-700 border-orange-200",
            rose: "bg-rose-100 text-rose-700 border-rose-200",
            cyan: "bg-cyan-100 text-cyan-700 border-cyan-200",
        };

        return (
            <div className={`relative flex flex-col lg:flex-row items-center gap-8 lg:gap-16 group ${align === "right" ? "lg:flex-row-reverse" : ""}`}>
                {/* Visual Side */}
                <div className="flex-1 w-full">
                    <div className="relative rounded-2xl bg-gradient-to-br from-slate-50 to-white border border-slate-200 p-8 shadow-lg hover:shadow-xl transition-all duration-300 group-hover:-translate-y-1">
                        <div className={`absolute top-0 right-0 p-4 opacity-10 text-9xl leading-none font-bold select-none ${colors[color].split(' ')[1]}`}>
                            {phase}
                        </div>
                        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center text-3xl mb-6 ${colors[color]} border`}>
                            {icon}
                        </div>
                        <h4 className="text-xl font-bold text-slate-900 mb-2">Key Capabilities</h4>
                        <ul className="space-y-3">
                            {[1, 2, 3].map((i) => (
                                <li key={i} className="flex items-center gap-3 text-slate-600">
                                    <div className={`w-1.5 h-1.5 rounded-full ${colors[color].split(' ')[1].replace('text', 'bg')}`}></div>
                                    <div className="h-2 bg-slate-100 rounded w-3/4 animate-pulse"></div>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                {/* Connector Node */}
                <div className="hidden lg:flex flex-col items-center justify-center z-10 relative">
                    <div className={`w-12 h-12 rounded-full border-4 border-white shadow-xl flex items-center justify-center text-xl bg-gradient-to-br ${color === 'primary' ? 'from-blue-500 to-blue-600' : color === 'purple' ? 'from-purple-500 to-purple-600' : 'from-slate-700 to-slate-800'} text-white`}>
                        {phase}
                    </div>
                </div>

                {/* Content Side */}
                <div className={`flex-1 w-full ${align === "right" ? "lg:text-right" : "lg:text-left"} text-center`}>
                    <div className={`inline-block px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider mb-4 border ${colors[color]}`}>
                        Phase {phase}
                    </div>
                    <h3 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4 leading-tight group-hover:text-primary-600 transition-colors">
                        {title}
                    </h3>
                    <p className="text-lg text-slate-600 leading-relaxed">
                        {description}
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="overflow-hidden bg-slate-50 font-sans">
            <SEOHelmet
                title="Free Molecular Docking Online | AutoDock Vina Cloud Platform - BioDockify"
                description="Run AutoDock Vina molecular docking online free. Cloud-based drug discovery platform for M.Pharm & PhD students. No installation, 130 free credits monthly. AI-powered results explanation."
                keywords="molecular docking online free, autodock vina online, free molecular docking, protein ligand docking online, drug discovery software online, molecular docking for students, cloud based molecular docking"
                canonical="https://biodockify.com/"
                schema={schemaMarkup}
            />

            {/* Premium Hero Section */}
            <section className="relative pt-32 pb-24 lg:pt-48 lg:pb-40 overflow-hidden bg-slate-900">
                {/* Cinematic Background */}
                <div className="absolute inset-0 z-0 select-none">
                    <img
                        src="/assets/images/hero-molecular.png"
                        alt="Molecular Bonding"
                        className="w-full h-full object-cover opacity-40 mix-blend-screen scale-105 animate-slow-zoom"
                    />
                    <div className="absolute inset-0 bg-gradient-to-b from-slate-900/90 via-slate-900/50 to-slate-50"></div>
                </div>

                <div className="relative z-10 container mx-auto px-4 text-center">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800/50 border border-slate-700 backdrop-blur-md text-primary-400 text-sm font-medium mb-8 animate-fade-in-up">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary-500"></span>
                        </span>
                        v4.0.0 Now Live: 7-Phase Discovery Pipeline
                    </div>

                    <h1 className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tight text-white mb-8 leading-[1.1] drop-shadow-2xl">
                        Drug Discovery <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 via-purple-400 to-secondary-400">
                            Reimagined.
                        </span>
                    </h1>

                    <p className="text-xl md:text-2xl text-slate-300 max-w-3xl mx-auto mb-12 leading-relaxed font-light">
                        The world's first <span className="text-white font-medium">zero-setup</span> cloud platform for students & researchers.
                        Screen, Simulate, and Analyze molecules in minutes.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <Link to="/dock/new" className="w-full sm:w-auto px-8 py-4 bg-primary-600 hover:bg-primary-500 text-white text-lg font-bold rounded-xl shadow-[0_0_30px_rgba(59,130,246,0.3)] hover:shadow-[0_0_50px_rgba(59,130,246,0.5)] transition-all transform hover:-translate-y-1">
                            Start Free Simulation
                        </Link>
                        <Link to="/contact" className="w-full sm:w-auto px-8 py-4 bg-white/5 hover:bg-white/10 text-white backdrop-blur-md border border-white/10 text-lg font-bold rounded-xl transition-all">
                            Request Demo
                        </Link>
                    </div>

                    <p className="mt-8 text-slate-400 text-sm font-medium">
                        Looking for the Admin Panel? <Link to="/admin" className="text-primary-400 hover:underline">Sign In Here</Link>
                    </p>
                </div>
            </section>

            {/* Quick Stats Grid */}
            <section className="relative z-20 -mt-16 container mx-auto px-4 mb-24">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 lg:gap-8 bg-white/80 backdrop-blur-xl p-6 rounded-2xl shadow-2xl border border-white/20">
                    {[
                        { label: "Jobs Processed", value: 15000, suffix: "+" },
                        { label: "Active Researchers", value: 450, suffix: "+" },
                        { label: "Cloud Uptime", value: 99.9, suffix: "%" },
                        { label: "Data Secured", value: 100, suffix: "%" }
                    ].map((stat, i) => (
                        <div key={i} className="text-center p-4">
                            <div className="text-3xl lg:text-4xl font-black text-slate-900 mb-1">
                                <CountUp end={stat.value} suffix={stat.suffix} />
                            </div>
                            <div className="text-xs lg:text-sm text-slate-500 font-bold uppercase tracking-widest">{stat.label}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* 7-Phase Pipeline Section */}
            <section className="py-24 bg-white relative overflow-hidden">
                <div className="container mx-auto px-4">
                    <div className="text-center max-w-4xl mx-auto mb-20">
                        <h2 className="text-primary-600 font-bold tracking-widest uppercase text-sm mb-3">End-to-End Workflow</h2>
                        <h3 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6">The 7-Phase Discovery Engine</h3>
                        <p className="text-xl text-slate-600 leading-relaxed">
                            From simple docking to complex molecular dynamics and toxicity prediction.
                            We automate the entire 7-step pipeline used by top pharmaceutical labs.
                        </p>
                    </div>

                    <div className="relative max-w-6xl mx-auto">
                        {/* Vertical Connector Line */}
                        <div className="hidden lg:block absolute left-1/2 transform -translate-x-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-slate-200 via-primary-300 to-slate-200"></div>

                        <div className="space-y-24">
                            <PipelinePhase
                                phase="1"
                                title="High-Throughput Virtual Screening (HTVS)"
                                description="Parallelized molecular docking using AutoDock Vina 1.2.5. Screen multi-ligand libraries against your target receptor with microsecond latency."
                                icon="🧬"
                                color="primary"
                            />
                            <PipelinePhase
                                phase="2"
                                title="Molecular Dynamics (MD) Simulation"
                                description="Full-atomistic simulations powered by OpenMM. Assess the temporal stability of ligand-protein complexes in explicit solvent (1ns - 100ns)."
                                icon="⚡"
                                align="right"
                                color="purple"
                            />
                            <PipelinePhase
                                phase="3"
                                title="Trajectory Interactivome Analysis"
                                description="Deep dive into structural dynamics using MDAnalysis. Quantify RMSD stability, RMSF flexibility, and track hydrogen bond lifetimes."
                                icon="📊"
                                color="indigo"
                            />
                            <PipelinePhase
                                phase="4"
                                title="Binding Free Energy (MM-PBSA)"
                                description="Convert MD trajectories into quantitative binding affinity (ΔG). More accurate than docking scores, accounting for solvation and entropy."
                                icon="💎"
                                align="right"
                                color="emerald"
                            />
                            <PipelinePhase
                                phase="5"
                                title="Lead Ranking & Hit Selection"
                                description="Multi-criteria decision making. Filter candidates based on Stability (RMSD < 2Å), Affinity (< -9 kcal/mol), and interaction consistency."
                                icon="🏆"
                                color="orange"
                            />
                            <PipelinePhase
                                phase="6"
                                title="ADMET Prediction"
                                description="Predict drug-likeness and safety profiles. Screen for Blood-Brain Barrier (BBB) permeability, hepatotoxicity, and carcinogenicity early."
                                icon="🏥"
                                align="right"
                                color="rose"
                            />
                            <PipelinePhase
                                phase="7"
                                title="Consensus Scoring"
                                description="Combine Docking, MD, and ADMET scores into a single robust ranking. Eliminate false positives and identify the true best-in-class leads."
                                icon="🎯"
                                color="cyan"
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-20 bg-slate-900 relative overflow-hidden">
                <div className="absolute inset-0 bg-[url('/assets/images/grid-pattern.png')] opacity-10"></div>
                <div className="container mx-auto px-4 text-center relative z-10">
                    <h2 className="text-3xl md:text-5xl font-bold text-white mb-8">Ready to Accelerate Your Research?</h2>
                    <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-10">
                        Join 450+ researchers using BioDockify for their thesis and publications.
                        Start your first simulation in less than 2 minutes.
                    </p>
                    <Link to="/auth/signup" className="inline-block px-10 py-5 bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-500 hover:to-primary-400 text-white text-xl font-bold rounded-full shadow-2xl transform hover:scale-105 transition-all">
                        Create Free Account
                    </Link>
                    <p className="mt-6 text-slate-500 text-sm">No credit card required • 130 Free Credits/Month</p>
                </div>
            </section>
        </div>
    )
}
