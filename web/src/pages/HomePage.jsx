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
                "softwareVersion": "2.0",
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
                            v2.1 Now Live: AI Analysis & Blog
                        </div>

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
                </div>
            </section>

            {/* Video Modal */}


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

                        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100 p-8 mt-8 rounded-2xl text-center">
                            <p className="font-bold text-blue-900 mb-2 text-xl">🎯 Start Your Research Today</p>
                            <p className="text-blue-800 max-w-2xl mx-auto">
                                Join thousands of M.Pharm and PhD students worldwide who trust BioDockify for molecular docking online free. No installation, no credit card, no barriers.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section - Alternating Layout */}
            <section id="features" className="py-24 overflow-hidden">
                <div className="container mx-auto px-4">
                    <div className="text-center max-w-3xl mx-auto mb-20">
                        <h2 className="text-primary-600 font-bold tracking-wide uppercase text-sm mb-3">Why BioDockify?</h2>
                        <h3 className="text-3xl md:text-4xl font-bold text-slate-900">Enterprise-grade docking infrastructure</h3>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {/* Feature 1 */}
                        <div className="bg-white rounded-2xl p-6 shadow-lg border border-slate-100 hover:shadow-xl transition-all text-center group">
                            <div className="relative mb-6 mx-auto w-full max-w-[280px] h-48 overflow-hidden rounded-xl">
                                <div className="absolute inset-0 bg-primary-200/20 group-hover:bg-primary-200/30 transition-colors"></div>
                                <img
                                    src="/assets/images/dashboard-interface.png"
                                    alt="Cloud Scalability"
                                    className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
                                />
                            </div>
                            <div className="w-12 h-12 rounded-xl bg-primary-100 text-primary-600 flex items-center justify-center text-2xl mb-4 mx-auto">
                                ☁️
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 mb-3">Infinite Cloud Scalability</h3>
                            <p className="text-slate-600 text-sm leading-relaxed mb-4">
                                Forget queuing on local clusters. BioDockify leverages AWS Batch to spin up thousands of instances instantly.
                            </p>
                        </div>

                        {/* Feature 2 */}
                        <div className="bg-white rounded-2xl p-6 shadow-lg border border-slate-100 hover:shadow-xl transition-all text-center group">
                            <div className="relative mb-6 mx-auto w-full max-w-[280px] h-48 overflow-hidden rounded-xl">
                                <div className="absolute inset-0 bg-secondary-200/20 group-hover:bg-secondary-200/30 transition-colors"></div>
                                <img
                                    src="/assets/images/hero-molecular.png"
                                    alt="Format Support"
                                    className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
                                />
                            </div>
                            <div className="w-12 h-12 rounded-xl bg-secondary-100 text-secondary-600 flex items-center justify-center text-2xl mb-4 mx-auto">
                                🧬
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 mb-3">Universal Format Support</h3>
                            <p className="text-slate-600 text-sm leading-relaxed mb-4">
                                We handle the messy work of file conversion. Upload PDB, SDF, MOL2, or SMILES and we automate the rest.
                            </p>
                        </div>

                        {/* Feature 3 */}
                        <div className="bg-white rounded-2xl p-6 shadow-lg border border-slate-100 hover:shadow-xl transition-all text-center group">
                            <div className="relative mb-6 mx-auto w-full max-w-[280px] h-48 overflow-hidden rounded-xl">
                                <div className="absolute inset-0 bg-teal-200/20 group-hover:bg-teal-200/30 transition-colors"></div>
                                <img
                                    src="/assets/images/dashboard-interface.png"
                                    alt="Analysis"
                                    className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
                                />
                            </div>
                            <div className="w-12 h-12 rounded-xl bg-teal-100 text-teal-600 flex items-center justify-center text-2xl mb-4 mx-auto">
                                🔬
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 mb-3">Interactive 3D Analysis</h3>
                            <p className="text-slate-600 text-sm leading-relaxed mb-4">
                                Visualize your docking results directly in the browser. No need to download massive files or install complex software.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Combined Testimonials & FAQ Section */}
            <section className="py-24 bg-slate-50 border-t border-slate-200">
                <div className="container mx-auto px-4">
                    <div className="grid lg:grid-cols-2 gap-16 items-start">
                        {/* Left Column: Testimonials */}
                        <div>
                            <div className="mb-10">
                                <h2 className="text-primary-600 font-bold tracking-wide uppercase text-sm mb-3">Community Love</h2>
                                <h3 className="text-3xl font-bold text-slate-900">Trusted by Researchers</h3>
                            </div>

                            <div className="space-y-6">
                                {[
                                    {
                                        name: "Sarah Jenkins",
                                        role: "M.Pharm Student",
                                        university: "University of Manchester",
                                        content: "BioDockify saved my thesis! I spent weeks trying to install AutoDock Vina on my laptop with no luck. With BioDockify, I was running simulations in minutes.",
                                        initial: "S"
                                    },
                                    {
                                        name: "Dr. Michael Chen",
                                        role: "Postdoctoral Researcher",
                                        university: "Stanford University",
                                        content: "The cloud scalability is a game changer. I screened 5,000 compounds in a single afternoon using the batch processing feature.",
                                        initial: "M"
                                    },
                                    {
                                        name: "Prof. Emily Rodriguez",
                                        role: "Computational Chemistry Dept.",
                                        university: "University of Toronto",
                                        content: "A fantastic teaching tool. My students can focus on drug design concepts instead of fighting with Linux command lines.",
                                        initial: "E"
                                    }
                                ].map((testimonial, i) => (
                                    <div key={i} className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-all">
                                        <div className="flex items-center gap-1 text-yellow-400 mb-3">
                                            {[...Array(5)].map((_, j) => (
                                                <svg key={j} className="w-4 h-4 fill-current" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path></svg>
                                            ))}
                                        </div>
                                        <p className="text-slate-700 mb-4 text-sm leading-relaxed italic">"{testimonial.content}"</p>
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-500 to-secondary-500 flex items-center justify-center text-white font-bold text-sm">
                                                {testimonial.initial}
                                            </div>
                                            <div>
                                                <div className="font-bold text-slate-900 text-sm">{testimonial.name}</div>
                                                <div className="text-xs text-slate-500">{testimonial.role}</div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
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
                                        q: "Is BioDockify really free for students?",
                                        a: "Yes! BioDockify offers 130 free credits monthly for all users, which is enough for dozens of molecular docking simulations."
                                    },
                                    {
                                        q: "What file formats are supported?",
                                        a: "We support PDBQT, PDB, SDF, MOL2, and SMILES formats with automatic conversion. Upload any format and we'll handle it."
                                    },
                                    {
                                        q: "How accurate is the docking?",
                                        a: "We use official AutoDock Vina 1.2.5, providing identical accuracy to local installations. Results are publication-ready."
                                    },
                                    {
                                        q: "Can I use this for my thesis?",
                                        a: "Absolutely! BioDockify is designed for academic research. You get AI-powered explanations and PDF reports for your thesis."
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
        </div >
    )
}
