import { Link } from 'react-router-dom'
import SEOHelmet from '../components/SEOHelmet'

export default function AboutPage() {
    const schemaMarkup = {
        "@context": "https://schema.org",
        "@type": "AboutPage",
        "mainEntity": {
            "@type": "Organization",
            "name": "BioDockify",
            "description": "Cloud-native molecular docking platform for drug discovery",
            "foundingDate": "2024",
            "url": "https://biodockify.com",
            "sameAs": [
                "https://twitter.com/biodockify",
                "https://github.com/biodockify",
                "https://linkedin.com/company/biodockify"
            ]
        }
    }

    return (
        <>
            <SEOHelmet
                title="About BioDockify | Cloud Molecular Docking Platform"
                description="Learn about BioDockify's mission to democratize drug discovery. Cloud-based AutoDock Vina platform for students, researchers, and biotech startups."
                keywords="about biodockify, molecular docking platform, cloud drug discovery, computational chemistry team, biotech startup tools"
                canonical="https://biodockify.com/about"
                schema={schemaMarkup}
            />

            <div className="min-h-screen bg-slate-50 pt-32 pb-20">
                <div className="container mx-auto px-4">
                    {/* Hero Section */}
                    <div className="max-w-3xl mx-auto text-center mb-16">
                        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6 tracking-tight">
                            Democratizing Drug Discovery
                        </h1>
                        <p className="text-xl text-slate-600 leading-relaxed">
                            We believe that access to high-performance computational tools shouldn't be limited by budget or infrastructure.
                        </p>
                    </div>

                    {/* Main Content */}
                    <div className="max-w-4xl mx-auto space-y-12">
                        {/* Our Story */}
                        <section className="bg-white p-8 md:p-12 rounded-2xl shadow-sm border border-slate-200">
                            <h2 className="text-2xl font-bold text-slate-900 mb-6">Our Story</h2>
                            <div className="prose prose-slate max-w-none text-slate-700 space-y-4">
                                <p>
                                    BioDockify was born from a simple frustration: <strong>molecular docking is too hard to set up and too slow to run locally.</strong>
                                </p>
                                <p>
                                    For decades, computational drug discovery was the domain of well-funded pharmaceutical companies and elite universities with massive high-performance computing (HPC) clusters. Students, independent researchers, and early-stage biotech startups were left behind, struggling with complex command-line tools, dependency hell, and days-long wait times for simple simulations.
                                </p>
                                <p>
                                    We built BioDockify to change that. By wrapping the industry-standard <strong>AutoDock Vina</strong> engine in a modern, cloud-native infrastructure, we've made virtual screening accessible to everyone. No Linux command line, no installation, no expensive hardware—just upload your molecules and get results in minutes.
                                </p>
                            </div>
                        </section>

                        {/* Our Technology */}
                        <section className="grid md:grid-cols-2 gap-8">
                            <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
                                <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center text-2xl mb-6">
                                    ☁️
                                </div>
                                <h3 className="text-xl font-bold text-slate-900 mb-4">Cloud-Native Power</h3>
                                <p className="text-slate-600 leading-relaxed">
                                    Our platform runs on AWS serverless infrastructure, allowing us to scale from one job to thousands instantly. We use parallel processing to cut docking times from hours to minutes, enabling high-throughput screening without the supercomputer price tag.
                                </p>
                            </div>
                            <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
                                <div className="w-12 h-12 rounded-xl bg-purple-50 text-purple-600 flex items-center justify-center text-2xl mb-6">
                                    🤖
                                </div>
                                <h3 className="text-xl font-bold text-slate-900 mb-4">AI-Enhanced Analysis</h3>
                                <p className="text-slate-600 leading-relaxed">
                                    Raw data is useless without insight. We integrate advanced AI models to explain binding affinities, analyze protein-ligand interactions, and provide educational context, making complex results understandable for students and experts alike.
                                </p>
                            </div>
                        </section>

                        {/* For Students & Researchers */}
                        <section className="bg-gradient-to-br from-slate-900 to-slate-800 p-8 md:p-12 rounded-2xl shadow-lg text-white">
                            <h2 className="text-2xl font-bold mb-6">Who We Serve</h2>
                            <div className="grid md:grid-cols-2 gap-8">
                                <div>
                                    <h3 className="text-lg font-bold text-primary-400 mb-2">🎓 M.Pharm & PhD Students</h3>
                                    <p className="text-slate-300 text-sm leading-relaxed">
                                        Focus on your research, not IT troubleshooting. Get publication-ready data, visualizations, and methodology reports instantly. Our free tier is designed specifically for academic learning.
                                    </p>
                                </div>
                                <div>
                                    <h3 className="text-lg font-bold text-primary-400 mb-2">🔬 Biotech Startups</h3>
                                    <p className="text-slate-300 text-sm leading-relaxed">
                                        Accelerate your lead discovery phase. Screen thousands of compounds cost-effectively before investing in expensive wet-lab experiments. Pay only for what you use.
                                    </p>
                                </div>
                            </div>
                        </section>

                        {/* Call to Action */}
                        <div className="text-center pt-8">
                            <p className="text-slate-600 mb-6">Ready to accelerate your research?</p>
                            <Link
                                to="/dock/new"
                                className="inline-flex items-center justify-center px-8 py-4 text-lg font-bold text-white transition-all duration-200 bg-primary-600 rounded-full hover:bg-primary-700 hover:shadow-lg hover:shadow-primary-500/30 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                            >
                                Start Docking for Free
                                <svg className="w-5 h-5 ml-2 -mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                </svg>
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
