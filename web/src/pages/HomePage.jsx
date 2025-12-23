import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import SEOHelmet from '../components/SEOHelmet'
import {
    Activity, GraduationCap, Factory, FlaskConical, Stethoscope,
    ArrowRight, CheckCircle2, Zap, Search, Globe, ShieldCheck, Layers,
    Brain, Package, BarChart3, TrendingUp, FileText, CheckCircle, Play, Server
} from 'lucide-react'
import heroImage from '../assets/images/hero-molecular-v36.png'
import bridgingGapImage from '../assets/images/bridging-gap-v36.png'

export default function HomePage() {
    const [isVisible, setIsVisible] = useState(false)

    useEffect(() => {
        setIsVisible(true)
    }, [])

    return (
        <div className="overflow-hidden bg-slate-50 font-sans text-slate-900">
            <SEOHelmet
                title="BioDockify | Intelligent Molecular Research Platform"
                description="Accelerating Drug Discovery Through Intelligent Molecular Research. Simplify and accelerate modern drug discovery research without expensive infrastructure."
                keywords="drug discovery, molecular analysis, computational pharmacology, virtual screening, medicinal chemistry"
                canonical="https://biodockify.com/"
            />

            {/* 1. CLOUD-NATIVE HERO SECTION (Restored) */}
            <section className="relative bg-slate-900 overflow-hidden pt-20 pb-32 lg:pt-32 lg:pb-40">
                {/* Background Glows */}
                <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-indigo-600/20 rounded-full blur-[120px] -translate-y-1/2 translate-x-1/3"></div>
                <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-emerald-600/10 rounded-full blur-[120px] translate-y-1/2 -translate-x-1/3"></div>

                <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="lg:grid lg:grid-cols-2 gap-16 items-center">

                        {/* LEFT: Text Content */}
                        <div className="max-w-2xl mb-12 lg:mb-0">
                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-900/50 border border-indigo-700/50 text-indigo-300 text-[10px] font-bold uppercase tracking-wider mb-8 animate-fade-in">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                                Major Release v6.0.0: Privacy-First Consensus Docking
                            </div>

                            <h1 className="text-5xl lg:text-7xl font-bold text-white mb-6 leading-[1.1] tracking-tight">
                                Cloud-Native, <br />
                                <span className="text-indigo-400">End-to-End</span> <br />
                                Drug Discovery.
                            </h1>

                            <p className="text-lg text-slate-400 mb-10 leading-relaxed max-w-lg">
                                Accelerate your pipeline from Virtual Screening to Lead Optimization with our high-performance cloud infrastructure.
                                Scalable, secure, and ready for production.
                            </p>

                            <div className="flex flex-wrap gap-4">
                                <Link to="/dock/new" className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white font-bold rounded-xl shadow-lg shadow-indigo-900/20 transition-all hover:-translate-y-0.5">
                                    <Play size={20} fill="currentColor" />
                                    <span>Start Free Trial</span>
                                </Link>
                                <Link to="/contact" className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-slate-800/50 hover:bg-slate-800 text-white font-semibold rounded-xl border border-slate-700 transition-all group">
                                    <span>Contact Sales</span>
                                    <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
                                </Link>
                            </div>
                        </div>

                        {/* RIGHT: Visual */}
                        <div className="relative animate-fade-in-up delay-200">
                            <div className="relative rounded-3xl overflow-hidden border border-slate-700/50 shadow-2xl bg-slate-800/50 backdrop-blur-sm group">
                                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-1000"></div>

                                {/* Floating Badge */}
                                <div className="absolute top-8 left-8 z-20 bg-slate-900/90 backdrop-blur-md p-4 rounded-xl border border-slate-700 shadow-xl flex items-center gap-4 max-w-[240px]">
                                    <div className="p-2 bg-emerald-500/20 rounded-lg">
                                        <Server size={24} className="text-emerald-400" />
                                    </div>
                                    <div>
                                        <div className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Secure Storage</div>
                                        <div className="text-sm font-bold text-white">Encrypted & Compliant</div>
                                    </div>
                                </div>

                                <img
                                    src={heroImage}
                                    alt="BioDockify Interface"
                                    className="w-full h-auto transform transition-transform duration-700 hover:scale-105"
                                    onError={(e) => {
                                        console.error("Hero image failed to load, falling back to placeholder");
                                        e.target.onerror = null;
                                        e.target.src = "https://images.unsplash.com/photo-1614935151651-0bea6508db6b?auto=format&fit=crop&q=80&w=1200";
                                    }}
                                />

                                {/* Overlay Text on Image (Bottom Right) */}
                                <div className="absolute bottom-0 right-0 p-8 text-right bg-gradient-to-t from-slate-900 via-slate-900/50 to-transparent w-full">
                                    <div className="text-2xl font-bold text-white tracking-widest uppercase opacity-80">Biotech Innovations</div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </section>

            {/* 2. 9-PHASE DISCOVERY ENGINE (WORKFLOW) */}
            <section id="workflow" className="py-32 bg-slate-50 relative overflow-hidden">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-24">
                        <h2 className="text-sm font-bold text-indigo-600 tracking-[0.2em] uppercase mb-4">The Enterprise Workflow</h2>
                        <h3 className="text-4xl md:text-6xl font-bold text-slate-900 mb-6">9-Phase Discovery Engine</h3>
                        <p className="text-xl text-slate-600 max-w-3xl mx-auto">
                            Automate the complex biology so your team can focus on the discovery.
                            A complete pipeline from target identification to FDA-compliant submission.
                        </p>
                    </div>

                    <div className="relative">
                        {/* Vertical Timeline Line */}
                        <div className="hidden lg:block absolute left-1/2 transform -translate-x-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-slate-200 via-indigo-200 to-slate-200"></div>

                        <div className="space-y-4 lg:space-y-24">
                            {/* 1. Target Identification */}
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

                            {/* 2. High-Throughput Docking */}
                            <TimelineItem
                                number="2"
                                title="High-Throughput Docking"
                                subtitle="AutoDock Vina Implementation"
                                description="Parallelized virtual screening of multi-ligand libraries. Generates ranked binding poses and affinity scores instantly."
                                tags={["Cloud Scaling", "PDBQT Automation", "Blind Docking"]}
                                status="completed"
                                side="right"
                                icon={<Package size={24} />}
                            />

                            {/* 3. Molecular Dynamics */}
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

                            {/* 4. Trajectory Analysis */}
                            <TimelineItem
                                number="4"
                                title="Trajectory Analysis"
                                subtitle="MDAnalysis Integration"
                                description="Quantify RMSD stability, RMSF flexibility, and Radius of Gyration. Identify true binders vs. sticky artifacts."
                                tags={["RMSD/RMSF", "H-Bond Lifetime", "Clustering"]}
                                status="completed"
                                side="right"
                                icon={<BarChart3 size={24} />}
                            />

                            {/* 5. Binding Free Energy */}
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

                            {/* 6. Lead Ranking */}
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

                            {/* 7. ADMET Profiling */}
                            <TimelineItem
                                number="7"
                                title="ADMET Profiling"
                                subtitle="RDKit Cheminformatics"
                                description="Screen for drug-likeness (Lipinski), toxicity alerts, and pharmacokinetic properties (absorption, distribution, metabolism)."
                                tags={["BBB Permeability", "Toxicity", "Oral Bioavailability"]}
                                status="completed"
                                side="left"
                                icon={<TrendingUp size={24} />}
                            />

                            {/* 8. Accuracy Benchmarking */}
                            <TimelineItem
                                number="8"
                                title="Accuracy Benchmarking"
                                subtitle="Validation & Quality Control"
                                description="Validate model performance against known experimental data (PDBbind, ChEMBL). Calculate R² and RMSE in real-time."
                                tags={["Correlation Analysis", "Scatter Plots", "Model Validation"]}
                                status="completed"
                                side="right"
                                icon={<CheckCircle2 size={24} />}
                            />

                            {/* 9. Regulatory Submission */}
                            <TimelineItem
                                number="9"
                                title="Regulatory Submission"
                                subtitle="FDA 21 CFR Part 11 Compliance"
                                description="Generate immutable audit logs and comprehensive PDF reports. Ready for internal review or regulatory submission."
                                tags={["Immutable Logs", "Role-Based Access", "Digital Compliance"]}
                                status="completed"
                                side="left"
                                icon={<FileText size={24} />}
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* 3. WHY BIODOCKIFY (Challenges & Advantages) */}
            <section className="py-24 bg-white border-t border-slate-100">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="lg:grid lg:grid-cols-2 gap-16 items-center">
                        <div className="mb-12 lg:mb-0">
                            <h2 className="text-sm font-bold text-indigo-600 tracking-widest uppercase mb-3">Why BioDockify?</h2>
                            <h3 className="text-3xl md:text-5xl font-bold text-slate-900 mb-6">Bridging the Gap Between Research and Discovery</h3>
                            <p className="text-lg text-slate-600 mb-8 leading-relaxed">
                                Drug discovery today faces major challenges—high costs, long timelines, fragmented tools, and limited accessibility.
                                BioDockify addresses these by providing a unified environment where scientists can focus on discovery rather than software complexity.
                            </p>
                            <div className="space-y-6">
                                <div className="flex items-start gap-4">
                                    <div className="p-2 bg-emerald-100 rounded-lg text-emerald-600 mt-1"><CheckCircle2 size={20} /></div>
                                    <div>
                                        <h4 className="font-bold text-lg text-slate-900">Faster Molecular Analysis</h4>
                                        <p className="text-slate-500">Rapid compound evaluation and screening.</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="p-2 bg-indigo-100 rounded-lg text-indigo-600 mt-1"><Globe size={20} /></div>
                                    <div>
                                        <h4 className="font-bold text-lg text-slate-900">Accessible Anywhere</h4>
                                        <p className="text-slate-500">Cloud-native with no hardware dependency.</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="p-2 bg-amber-100 rounded-lg text-amber-600 mt-1"><Zap size={20} /></div>
                                    <div>
                                        <h4 className="font-bold text-lg text-slate-900">For All Skill Levels</h4>
                                        <p className="text-slate-500">Designed for both beginners and advanced researchers.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {/* 3D Visual for Section 3 */}
                        <div className="relative">
                            <div className="absolute inset-0 bg-blue-600 blur-[80px] opacity-10 rounded-full"></div>
                            <img
                                src={bridgingGapImage}
                                alt="BioDockify Bridging Research and Discovery"
                                className="relative w-full h-auto rounded-3xl shadow-2xl border border-slate-200 transform hover:scale-[1.02] transition-transform duration-500"
                                onError={(e) => {
                                    e.target.onerror = null;
                                    e.target.src = "https://placehold.co/800x600/f8fafc/e2e8f0?text=Analysis+Interface";
                                }}
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* 4. WHO CAN USE (Personas) */}
            <section className="py-24 bg-slate-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">Who Can Use BioDockify?</h2>
                        <p className="text-lg text-slate-600">Start-to-finish support for the entire scientific community.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <PersonaCard
                            icon={<GraduationCap className="w-8 h-8 text-indigo-600" />}
                            title="PhD Scholars & Academia"
                            desc="Conduct computational studies, validate hypotheses, and support publications with reliable molecular insights."
                            color="indigo"
                        />
                        <PersonaCard
                            icon={<Factory className="w-8 h-8 text-emerald-600" />}
                            title="Pharma & Biotech"
                            desc="Accelerate lead identification and reduce early-stage R&D timelines with scalable screening."
                            color="emerald"
                        />
                        <PersonaCard
                            icon={<FlaskConical className="w-8 h-8 text-violet-600" />}
                            title="CROs & Research Labs"
                            desc="Perform high-throughput screening and molecular analysis for multiple clients efficiently."
                            color="violet"
                        />
                        <PersonaCard
                            icon={<Stethoscope className="w-8 h-8 text-blue-600" />}
                            title="Medicinal Chemistry"
                            desc="Explore compound behavior, binding potential, and structure-activity relationships."
                            color="blue"
                        />
                    </div>
                </div>
            </section>

            {/* 5. FINAL CTA */}
            <section className="py-32 bg-[#0B1121] text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-10"></div>
                <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] bg-indigo-600/20 rounded-full blur-[120px]"></div>
                <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-emerald-600/20 rounded-full blur-[120px]"></div>

                <div className="max-w-4xl mx-auto px-4 relative z-10">
                    <h2 className="text-5xl font-bold text-white mb-8">Start Your Research with Confidence</h2>
                    <p className="text-xl text-slate-300 mb-12 max-w-2xl mx-auto leading-relaxed">
                        Reduce experimental risk, improve decision-making, and accelerate discovery timelines.
                        From academic exploration to industry-ready insights.
                    </p>
                    <div className="flex flex-col sm:flex-row justify-center gap-6">
                        <Link to="/dock/new" className="px-10 py-5 bg-white text-slate-900 font-bold text-lg rounded-2xl shadow-xl hover:bg-indigo-50 transition-all transform hover:-translate-y-1">
                            Begin Your Research
                        </Link>
                        <Link to="/contact" className="px-10 py-5 bg-indigo-900/50 text-white font-bold text-lg rounded-2xl border border-indigo-700/50 hover:bg-indigo-800 transition-all backdrop-blur-sm">
                            Contact Us
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    )
}

// --- Subcomponents ---

function TimelineItem({ number, title, subtitle, description, tags, status, side, icon }) {
    const isCompleted = status === 'completed';
    const alignClass = side === 'left' ? 'lg:flex-row' : 'lg:flex-row-reverse';
    const textClass = side === 'left' ? 'lg:text-right' : 'lg:text-left';
    const paddingClass = side === 'left' ? 'lg:pr-20' : 'lg:pl-20';
    const colorClass = isCompleted ? 'border-emerald-500' : 'border-indigo-400';
    const bgIconClass = isCompleted ? 'bg-emerald-100 text-emerald-600' : 'bg-indigo-100 text-indigo-600';

    return (
        <div className={`relative flex flex-col ${alignClass} items-center group`}>
            {/* Center Node (Desktop only) */}
            <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center justify-center w-14 h-14 rounded-full bg-white border-4 border-slate-100 z-10 shadow-xl group-hover:scale-110 transition-transform hidden lg:flex">
                <span className={`text-xl font-bold ${isCompleted ? 'text-emerald-600' : 'text-slate-400'}`}>{number}</span>
            </div>

            {/* Content Card */}
            <div className={`w-full lg:w-1/2 ${paddingClass} ${textClass}`}>
                <div className={`bg-white p-8 rounded-3xl shadow-sm border border-slate-100 hover:shadow-2xl hover:border-indigo-100 transition-all duration-300 transform group-hover:-translate-y-1 relative overflow-hidden`}>
                    {/* Decorative top border */}
                    <div className={`absolute top-0 left-0 w-full h-1 ${isCompleted ? 'bg-emerald-500' : 'bg-indigo-500'}`}></div>

                    <div className={`flex items-center gap-4 mb-4 ${side === 'left' ? 'lg:justify-end' : 'lg:justify-start'}`}>
                        {side === 'right' && <div className={`p-3 rounded-2xl ${bgIconClass}`}>{icon}</div>}
                        <div>
                            <h3 className="text-xl font-bold text-slate-900">{title}</h3>
                            <div className="text-xs font-bold text-indigo-600 uppercase tracking-widest">{subtitle}</div>
                        </div>
                        {side === 'left' && <div className={`p-3 rounded-2xl ${bgIconClass}`}>{icon}</div>}
                    </div>

                    <p className="text-slate-600 mb-6 leading-relaxed font-medium">{description}</p>

                    <div className={`flex flex-wrap gap-2 ${side === 'left' ? 'lg:justify-end' : 'lg:justify-start'}`}>
                        {tags.map((tag, i) => (
                            <span key={i} className="px-3 py-1 bg-slate-50 text-slate-500 text-xs font-bold rounded-lg border border-slate-100">
                                {tag}
                            </span>
                        ))}
                        <span className="px-3 py-1 bg-emerald-100 text-emerald-700 text-xs font-bold rounded-lg flex items-center gap-1">
                            <CheckCircle size={12} /> Live
                        </span>
                    </div>
                </div>
            </div>

            {/* Empty Space for alignment */}
            <div className="hidden lg:block lg:w-1/2"></div>
        </div>
    )
}

function PersonaCard({ icon, title, desc, color }) {
    const bgColors = {
        indigo: 'bg-indigo-50 hover:bg-indigo-100',
        emerald: 'bg-emerald-50 hover:bg-emerald-100',
        violet: 'bg-violet-50 hover:bg-violet-100',
        blue: 'bg-blue-50 hover:bg-blue-100'
    }

    return (
        <div className={`p-8 rounded-2xl ${bgColors[color]} transition-colors text-center border border-slate-100`}>
            <div className="w-16 h-16 mx-auto bg-white rounded-full flex items-center justify-center shadow-sm mb-6">
                {icon}
            </div>
            <h3 className="text-lg font-bold text-slate-900 mb-3">{title}</h3>
            <p className="text-slate-600 text-sm leading-relaxed">{desc}</p>
        </div>
    )
}
