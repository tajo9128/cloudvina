import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import SEOHelmet from '../components/SEOHelmet'
import {
    PlayCircle, Zap, Activity, CheckCircle, Clock,
    ArrowRight, Github, Info, Download,
    Box, Cpu, FileCode, FlaskConical, LineChart, Cloud,
    Brain, GraduationCap, Factory, Search, Database, CreditCard
} from 'lucide-react'
import CountUp from '../components/CountUp'

export default function HomePage() {
    const [isVisible, setIsVisible] = useState(false)

    useEffect(() => {
        setIsVisible(true)
    }, [])

    const schemaMarkup = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "SoftwareApplication",
                "name": "BioDockify",
                "applicationCategory": "ScientificApplication",
                "operatingSystem": "Web Browser",
                "offers": {
                    "@type": "AggregateOffer",
                    "lowPrice": "49",
                    "priceCurrency": "USD"
                },
                "featureList": [
                    "AutoDock Vina Integration",
                    "Cloud-based Molecular Docking",
                    "MD Simulation via OpenMM",
                    "Enterprise Drug Discovery"
                ],
                "softwareVersion": "6.0.0"
            }
        ]
    }

    return (
        <div className="overflow-hidden bg-slate-50 font-sans text-slate-900">
            <SEOHelmet
                title="BioDockify | Enterprise Cloud Drug Discovery"
                description="Run AutoDock Vina & OpenMM simulations on high-performance cloud infrastructure. Scalable drug discovery platform for biotechs and academic labs."
                keywords="molecular docking, cloud computing, drug discovery software, enterprise bioinformatics, virtual screening"
                canonical="https://biodockify.com/"
                schema={schemaMarkup}
            />

            {/* Hero Section */}
            <section className="relative pt-32 pb-24 lg:pt-48 lg:pb-32 bg-[#0B1121] overflow-hidden">
                {/* Abstract Background Elements */}
                <div className="absolute inset-0 z-0">
                    <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-indigo-950 via-slate-900 to-black opacity-90"></div>
                    <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] bg-primary-600/10 rounded-full blur-[120px]"></div>
                    <div className="absolute bottom-[-10%] left-[-5%] w-[500px] h-[500px] bg-purple-600/10 rounded-full blur-[120px]"></div>
                    {/* Grid Pattern */}
                    <div className="absolute inset-0 bg-[url('/assets/images/grid.svg')] opacity-[0.05]"></div>
                </div>

                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="lg:grid lg:grid-cols-2 lg:gap-16 items-center">
                        <div className="text-left mb-12 lg:mb-0">
                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-bold uppercase tracking-wider mb-6 animate-fade-in">
                                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                                Major Release v6.0.0: Privacy-First Consensus Docking
                            </div>
                            <h1 className="text-5xl lg:text-7xl font-bold text-white mb-6 leading-tight tracking-tight">
                                Cloud-Native, <br />
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">End-to-End</span> <br />
                                Drug Discovery.
                            </h1>
                            <p className="text-xl text-slate-400 mb-8 leading-relaxed max-w-xl font-light">
                                Accelerate your pipeline from <strong>Virtual Screening</strong> to <strong>Lead Optimization</strong> with our high-performance cloud infrastructure. Scalable, secure, and ready for production.
                            </p>

                            <div className="flex flex-col sm:flex-row gap-4 mb-12">
                                <Link to="/dock/new" className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl transition-all shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:shadow-[0_0_30px_rgba(79,70,229,0.5)] transform hover:-translate-y-1">
                                    <PlayCircle size={20} fill="currentColor" className="text-white/20" />
                                    <span>Start Free Trial</span>
                                </Link>
                                <Link to="/contact" className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 text-white font-semibold rounded-xl backdrop-blur-sm transition-all">
                                    <span>Contact Sales</span>
                                    <ArrowRight size={18} />
                                </Link>
                            </div>

                            {/* Status Pill */}
                            <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-md max-w-sm">
                                <div className="flex items-center justify-between mb-3">
                                    <span className="text-slate-200 text-sm font-medium flex items-center gap-2">
                                        <Activity size={16} className="text-emerald-400" /> Platform Status
                                    </span>
                                    <span className="text-xs font-mono text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded">99.9% UPTIME</span>
                                </div>
                                <div className="w-full bg-slate-800 rounded-full h-2 mb-2">
                                    <div className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full w-[85%]" title="Pipeline Optimization"></div>
                                </div>
                                <div className="flex justify-between text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                                    <span>Screening</span>
                                    <span>MD Sim</span>
                                    <span>Analysis</span>
                                    <span className="text-slate-700">Report</span>
                                </div>
                            </div>
                        </div>

                        {/* Hero Right: 3D Molecule or Abstract Vis */}
                        <div className="hidden lg:block relative">
                            <div className="absolute inset-0 bg-indigo-500/20 blur-[100px] rounded-full"></div>
                            <img
                                src="/assets/images/hero-molecular.png"
                                alt="Molecular Dynamics Visualization"
                                className="relative z-10 w-full h-auto rounded-2xl shadow-2xl border border-white/10 mix-blend-screen opacity-90 animate-float"
                            />

                            {/* Floating Cards */}
                            <div className="absolute top-10 -left-10 bg-slate-900/90 backdrop-blur-xl p-4 rounded-xl border border-slate-700 shadow-xl animate-fade-in-up delay-100">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-green-500/20 rounded-lg text-green-400"><Database size={20} /></div>
                                    <div>
                                        <div className="text-xs text-slate-400 uppercase font-bold">Secure Storage</div>
                                        <div className="text-white font-bold">Encrypted & Compliant</div>
                                    </div>
                                </div>
                            </div>

                            <div className="absolute bottom-10 -right-5 bg-slate-900/90 backdrop-blur-xl p-4 rounded-xl border border-slate-700 shadow-xl animate-fade-in-up delay-300">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-purple-500/20 rounded-lg text-purple-400"><Zap size={20} /></div>
                                    <div>
                                        <div className="text-xs text-slate-400 uppercase font-bold">High Performance</div>
                                        <div className="text-white font-bold">GPU Accelerated</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Quick Stats Strip */}
            <div className="bg-[#0f172a] border-y border-slate-800">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-slate-800/50">
                        {[
                            { label: "Compounds Screened", value: 1500, icon: <CheckCircle size={18} className="text-green-500" /> },
                            { label: "Active Organizations", value: 124, icon: <Factory size={18} className="text-purple-500" /> },
                            { label: "Compute Hours", value: 850, icon: <Clock size={18} className="text-blue-500" /> },
                            { label: "Success Rate", value: 92, suffix: "%", icon: <LineChart size={18} className="text-yellow-500" /> }
                        ].map((stat, i) => (
                            <div key={i} className="py-8 px-4 text-center group hover:bg-slate-800/30 transition-colors">
                                <div className="flex items-center justify-center gap-2 text-slate-400 text-xs font-bold uppercase tracking-widest mb-2">
                                    {stat.icon} {stat.label}
                                </div>
                                <div className="text-3xl font-bold text-white">
                                    {stat.prefix}<CountUp end={stat.value} duration={2} suffix={stat.suffix || "+"} />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Pipeline Timeline Section */}
            <section id="pipeline" className="py-24 bg-slate-50 relative overflow-hidden">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-20">
                        <h2 className="text-base text-indigo-600 font-bold tracking-widest uppercase mb-2">The Enterprise Workflow</h2>
                        <h3 className="text-4xl md:text-5xl font-bold text-black mb-6">9-Phase Discovery Engine</h3>
                        <p className="text-xl text-slate-600 max-w-2xl mx-auto">
                            Automate the complex biology so your team can focus on the discovery.
                            From target identification to FDA-compliant submission.
                        </p>
                    </div>

                    <div className="relative">
                        {/* Vertical Timeline Line */}
                        <div className="hidden lg:block absolute left-1/2 transform -translate-x-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-slate-300 via-indigo-200 to-slate-200"></div>

                        <div className="space-y-24">
                            {/* Phase 1 */}
                            <TimelineItem
                                number="1"
                                title="Target Identification"
                                subtitle="AI-Driven Target Prediction"
                                description="Upload a FASTA sequence or PDB ID. Our AI models predict bindability and druggability scores before you start."
                                tags={["Deep Learning", "Binding Site Detection", "Hotspot Analysis"]}
                                status="completed"
                                side="left"
                                icon={<Brain size={24} />}
                            />

                            {/* Phase 2 (Old 1) */}
                            <TimelineItem
                                number="2"
                                title="High-Throughput Docking"
                                subtitle="AutoDock Vina Implementation"
                                description="Parallelized virtual screening of multi-ligand libraries. Generates ranked binding poses and affinity scores instantly."
                                tags={["Cloud Scaling", "PDBQT Automation", "Blind Docking"]}
                                status="completed"
                                side="right"
                                icon={<Box size={24} />}
                            />

                            {/* Phase 3 (Old 2) */}
                            <TimelineItem
                                number="3"
                                title="Molecular Dynamics"
                                subtitle="OpenMM Production Runs"
                                description="Full-atomistic simulations on dedicated GPU instances. Validate ligand stability with energy minimization and equilibration."
                                tags={["Explicit Solvent", "Amber14", "GPU Clusters"]}
                                status="completed"
                                side="left"
                                icon={<Zap size={24} />}
                            />

                            {/* Phase 4 (Old 3) */}
                            <TimelineItem
                                number="4"
                                title="Trajectory Analysis"
                                subtitle="MDAnalysis Integration"
                                description="Quantify RMSD stability, RMSF flexibility, and Radius of Gyration. Identify true binders vs. sticky artifacts."
                                tags={["RMSD/RMSF", "H-Bond Lifetime", "Clustering"]}
                                status="completed"
                                side="right"
                                icon={<LineChart size={24} />}
                            />

                            {/* Phase 5 (Old 4) */}
                            <TimelineItem
                                number="5"
                                title="Binding Free Energy"
                                subtitle="MM-PBSA / MM-GBSA"
                                description="Calculate rigorous binding free energies (ΔG) including solvation effects. More accurate ranking than simple docking scores."
                                tags={["Advanced Scoring", "Implicit Solvent", "Entropy Estimation"]}
                                status="completed"
                                side="left"
                                icon={<FlaskConical size={24} />}
                            />

                            {/* Phase 6 (Old 5) */}
                            <TimelineItem
                                number="6"
                                title="Lead Ranking"
                                subtitle="Consensus Scoring"
                                description="Normalize scores from Docking, MD, and GBSA. Filter false positives with a robust consensus algorithm."
                                tags={["Weighted Scoring", "Rank Aggregation", "Outlier Removal"]}
                                status="completed"
                                side="right"
                                icon={<Search size={24} />}
                            />

                            {/* Phase 7 (Old 6) */}
                            <TimelineItem
                                number="7"
                                title="ADMET Profiling"
                                subtitle="RDKit Cheminformatics"
                                description="Screen for drug-likeness (Lipinski), toxicity alerts, and pharmacokinetic properties (absorption, distribution, metabolism)."
                                tags={["BBB Permeability", "Toxicity", "Oral Bioavailability"]}
                                status="completed"
                                side="left"
                                icon={<Activity size={24} />}
                            />

                            {/* Phase 8 (New) */}
                            <TimelineItem
                                number="8"
                                title="Accuracy Benchmarking"
                                subtitle="Validation & Quality Control"
                                description="Validate model performance against known experimental data (PDBbind, ChEMBL). Calculate R² and RMSE in real-time."
                                tags={["Correlation Analysis", "Scatter Plots", "Model Validation"]}
                                status="completed"
                                side="right"
                                icon={<CheckCircle size={24} />}
                            />

                            {/* Phase 9 (Old 7 - Enhanced) */}
                            <TimelineItem
                                number="9"
                                title="Regulatory Submission"
                                subtitle="FDA 21 CFR Part 11 Compliance"
                                description="Generate immutable audit logs and comprehensive PDF reports. Ready for internal review or regulatory submission."
                                tags={["Immutable Logs", "Role-Based Access", "Digital Compliance"]}
                                status="completed"
                                side="left"
                                icon={<FileCode size={24} />}
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* 12-Week Roadmap Section */}
            <section className="py-24 bg-slate-900 text-white relative overflow-hidden">
                <div className="absolute inset-0 bg-[url('/assets/images/grid.svg')] opacity-[0.03]"></div>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="text-center mb-16">
                        <h2 className="text-base text-indigo-400 font-bold tracking-widest uppercase mb-2">Strategic Roadmap</h2>
                        <h3 className="text-3xl md:text-4xl font-bold text-white mb-6">12-Week Integration Plan</h3>
                        <p className="text-slate-400 max-w-2xl mx-auto">
                            Our structured approach ensures seamless adoption and scalability for your organization.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                        {/* W 1-4 */}
                        <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700 hover:border-indigo-500 transition-colors group">
                            <div className="flex items-center justify-between mb-4">
                                <div className="text-xs font-bold bg-slate-700 px-2 py-1 rounded text-slate-300">Weeks 1-4</div>
                                <CheckCircle className="w-5 h-5 text-emerald-500" />
                            </div>
                            <h4 className="text-xl font-bold mb-2 text-white group-hover:text-indigo-400 transition-colors">Foundation</h4>
                            <ul className="space-y-3 text-sm text-slate-400">
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-500"></div> User Onboarding & Analytics</li>
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-500"></div> High-Throughput Docking</li>
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-500"></div> MD Simulation Pipeline</li>
                            </ul>
                        </div>

                        {/* W 5-8 */}
                        <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700 hover:border-indigo-500 transition-colors group">
                            <div className="flex items-center justify-between mb-4">
                                <div className="text-xs font-bold bg-slate-700 px-2 py-1 rounded text-slate-300">Weeks 5-8</div>
                                <CheckCircle className="w-5 h-5 text-emerald-500" />
                            </div>
                            <h4 className="text-xl font-bold mb-2 text-white group-hover:text-indigo-400 transition-colors">Enterprise Core</h4>
                            <ul className="space-y-3 text-sm text-slate-400">
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-500"></div> FDA 21 CFR Part 11 Logs</li>
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-500"></div> Role-Based Access (RBAC)</li>
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-500"></div> Accuracy Benchmarking</li>
                            </ul>
                        </div>

                        {/* W 9-10 */}
                        <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700 hover:border-indigo-500 transition-colors group">
                            <div className="flex items-center justify-between mb-4">
                                <div className="text-xs font-bold bg-slate-700 px-2 py-1 rounded text-slate-300">Weeks 9-10</div>
                                <CheckCircle className="w-5 h-5 text-emerald-500" />
                            </div>
                            <h4 className="text-xl font-bold mb-2 text-white group-hover:text-indigo-400 transition-colors">Ecosystem</h4>
                            <ul className="space-y-3 text-sm text-slate-400">
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-500"></div> Python SDK Release</li>
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-500"></div> Developer API Portal</li>
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-500"></div> Custom Integrations</li>
                            </ul>
                        </div>

                        {/* W 11-12 */}
                        <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700 hover:border-indigo-500 transition-colors group">
                            <div className="flex items-center justify-between mb-4">
                                <div className="text-xs font-bold bg-indigo-900/50 px-2 py-1 rounded text-indigo-300 border border-indigo-500/30">Weeks 11-12</div>
                                <Clock className="w-5 h-5 text-indigo-400" />
                            </div>
                            <h4 className="text-xl font-bold mb-2 text-white group-hover:text-indigo-400 transition-colors">Scale & Optimization</h4>
                            <ul className="space-y-3 text-sm text-slate-400">
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-indigo-500"></div> Global GPU Auto-Scaling</li>
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-indigo-500"></div> Multi-Region Redundancy</li>
                                <li className="flex gap-2"><div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-indigo-500"></div> Dedicated Instance Pools</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </section>

            {/* Tools Grid Section */}
            <section className="py-24 bg-white">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold text-black mb-6">Powered by Industry Standards</h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                            <ToolCard name="AutoDock Vina" type="Docking Engine" icon={<Box />} desc="The industry standard for molecular docking." color="blue" />
                            <ToolCard name="OpenMM" type="MD Engine" icon={<Zap />} desc="High-performance toolkit for molecular simulation." color="purple" />
                            <ToolCard name="MDAnalysis" type="Analysis Library" icon={<LineChart />} desc="Advanced trajectory analysis for complex systems." color="orange" />
                            <ToolCard name="RDKit" type="Cheminformatics" icon={<FlaskConical />} desc="Robust cheminformatics and machine learning tools." color="green" />
                            <ToolCard name="FastAPI" type="Backend Framework" icon={<Cpu />} desc="High-performance, secure API infrastructure." color="indigo" />
                            <ToolCard name="Cloud Compute" type="Infrastructure" icon={<Cloud />} desc="Dedicated GPU clusters for maximum throughput." color="yellow" />
                        </div>
                    </div>
                </div>
            </section>

            {/* User Persona Section */}
            <section className="py-24 bg-slate-900 text-white">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                        <div className="text-center">
                            <div className="w-16 h-16 mx-auto bg-blue-500/20 rounded-2xl flex items-center justify-center text-blue-400 mb-6">
                                <Brain size={32} />
                            </div>
                            <h3 className="text-xl font-bold mb-3">Pharma R&D</h3>
                            <p className="text-slate-400">Optimize lead candidates with specialized Blood-Brain Barrier (BBB) permeability filtering.</p>
                        </div>
                        <div className="text-center">
                            <div className="w-16 h-16 mx-auto bg-emerald-500/20 rounded-2xl flex items-center justify-center text-emerald-400 mb-6">
                                <GraduationCap size={32} />
                            </div>
                            <h3 className="text-xl font-bold mb-3">Academic Labs</h3>
                            <p className="text-slate-400">Accelerate publication timelines with high-throughput simulation capabilities.</p>
                        </div>
                        <div className="text-center">
                            <div className="w-16 h-16 mx-auto bg-purple-500/20 rounded-2xl flex items-center justify-center text-purple-400 mb-6">
                                <Factory size={32} />
                            </div>
                            <h3 className="text-xl font-bold mb-3">Biotech Startups</h3>
                            <p className="text-slate-400">Rapidly screen thousands of compounds and fail fast with predictive toxicity ADMET.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Final CTA Section */}
            <section className="py-24 bg-indigo-600 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-1/2 h-full opacity-10 pointer-events-none mix-blend-screen">
                    <img
                        src="https://images.unsplash.com/photo-1559757175-5700dde675bc?auto=format&fit=crop&q=80&w=2000"
                        alt="White Ribbon Protein Structure"
                        className="w-full h-full object-cover"
                    />
                </div>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
                    <h2 className="text-4xl font-bold text-white mb-6">Ready to Accelerate Your Research?</h2>
                    <p className="text-xl text-indigo-100 max-w-2xl mx-auto mb-10">
                        Join the platform that is democratizing High-Performance Computing for biology.
                        Start your first docking run in minutes.
                    </p>
                    <div className="flex justify-center gap-4">
                        <Link to="/dock/new" className="px-8 py-4 bg-white text-indigo-600 font-bold rounded-xl shadow-xl hover:bg-indigo-50 transition-all transform hover:-translate-y-1">
                            Get Started Free
                        </Link>
                        <Link to="/contact" className="px-8 py-4 bg-indigo-700 text-white font-bold rounded-xl border border-indigo-500 hover:bg-indigo-800 transition-all">
                            Talk to Us
                        </Link>
                    </div>
                </div>
            </section>


        </div>
    )
}

function TimelineItem({ number, title, subtitle, description, tags, status, side, icon }) {
    const isCompleted = status === 'completed';
    const alignClass = side === 'left' ? 'lg:flex-row' : 'lg:flex-row-reverse';
    const textClass = side === 'left' ? 'lg:text-right' : 'lg:text-left';
    const paddingClass = side === 'left' ? 'lg:pr-16' : 'lg:pl-16';
    const colorClass = isCompleted ? 'border-emerald-500' : 'border-indigo-400';
    const bgIconClass = isCompleted ? 'bg-emerald-100 text-emerald-600' : 'bg-indigo-100 text-indigo-600';

    return (
        <div className={`relative flex flex-col ${alignClass} items-center group`}>
            {/* Center Node */}
            <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-12 h-12 rounded-full bg-white border-4 border-slate-100 z-10 shadow-lg group-hover:scale-110 transition-transform hidden lg:flex">
                <span className={`text-lg font-bold ${isCompleted ? 'text-emerald-600' : 'text-slate-400'}`}>{number}</span>
            </div>

            {/* Content Card */}
            <div className={`w-full lg:w-1/2 ${paddingClass} ${textClass}`}>
                <div className={`bg-white p-8 rounded-2xl shadow-sm border-l-4 ${colorClass} hover:shadow-xl transition-all duration-300 transform group-hover:-translate-y-1`}>
                    <div className={`flex items-center gap-4 mb-4 ${side === 'left' ? 'lg:justify-end' : 'lg:justify-start'}`}>
                        {side === 'right' && <div className={`p-3 rounded-xl ${bgIconClass}`}>{icon}</div>}
                        <div>
                            <h3 className="text-xl font-bold text-black">{title}</h3>
                            <div className="text-sm font-semibold text-indigo-600 uppercase tracking-wide">{subtitle}</div>
                        </div>
                        {side === 'left' && <div className={`p-3 rounded-xl ${bgIconClass}`}>{icon}</div>}
                    </div>

                    <p className="text-slate-600 mb-6 leading-relaxed">{description}</p>

                    <div className={`flex flex-wrap gap-2 ${side === 'left' ? 'lg:justify-end' : 'lg:justify-start'}`}>
                        {tags.map((tag, i) => (
                            <span key={i} className="px-3 py-1 bg-slate-100 text-slate-600 text-xs font-semibold rounded-full">
                                {tag}
                            </span>
                        ))}
                        {isCompleted ? (
                            <span className="px-3 py-1 bg-emerald-100 text-emerald-700 text-xs font-bold rounded-full flex items-center gap-1">
                                <CheckCircle size={12} /> Live
                            </span>
                        ) : (
                            <span className="px-3 py-1 bg-indigo-100 text-indigo-700 text-xs font-bold rounded-full flex items-center gap-1">
                                <Clock size={12} /> Coming Soon
                            </span>
                        )}
                    </div>
                </div>
            </div>

            {/* Empty Space for alignment */}
            <div className="hidden lg:block lg:w-1/2"></div>
        </div>
    )
}

function ToolCard({ name, type, icon, desc, color }) {
    const colors = {
        blue: "bg-blue-50 text-blue-600 border-blue-200 group-hover:border-blue-500",
        purple: "bg-purple-50 text-purple-600 border-purple-200 group-hover:border-purple-500",
        orange: "bg-orange-50 text-orange-600 border-orange-200 group-hover:border-orange-500",
        green: "bg-green-50 text-green-600 border-green-200 group-hover:border-green-500",
        indigo: "bg-indigo-50 text-indigo-600 border-indigo-200 group-hover:border-indigo-500",
        yellow: "bg-yellow-50 text-yellow-600 border-yellow-200 group-hover:border-yellow-500",
    }

    return (
        <div className={`p-6 rounded-2xl border ${colors[color].split(' ')[2]} ${colors[color].split(' ')[3]} bg-white shadow-sm hover:shadow-lg transition-all group text-left`}>
            <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${colors[color].split(' ')[0]} ${colors[color].split(' ')[1]}`}>
                {icon}
            </div>
            <h3 className="text-lg font-bold text-black">{name}</h3>
            <p className="text-xs font-bold uppercase tracking-wide text-slate-400 mb-3">{type}</p>
            <p className="text-slate-600 text-sm leading-relaxed mb-4">{desc}</p>
            <div className="flex items-center justify-between pt-4 border-t border-slate-100">
                <span className="text-xs font-bold text-slate-400">Enterprise Ready</span>
                <Link to="/pricing" className="text-xs font-bold text-indigo-600 bg-indigo-50 px-2 py-1 rounded-full hover:bg-indigo-100 transition-colors">View Pricing</Link>
            </div>
        </div>
    )
}
