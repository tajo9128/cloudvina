
import React from 'react';
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet-async'; // Assuming this is used, if not I'll just skip or use document.title
// This replaces the previous static version with dynamic GitHub release fetching
import {
    Download,
    WifiOff,
    ShieldCheck,
    Cpu,
    Activity,
    Layers,
    Terminal,
    CheckCircle,
    Monitor
} from 'lucide-react';

const DockingStudioPage = () => {
    const [downloadUrl, setDownloadUrl] = React.useState('https://github.com/tajo9128/Docking-studio/releases/latest');
    const [version, setVersion] = React.useState('v1.0.0');

    // SEO setup & Fetch Download Link
    React.useEffect(() => {
        document.title = "BioDockify Docking Studio - Secure Offline Molecular Docking Software";
        const metaDesc = document.querySelector('meta[name="description"]');
        if (metaDesc) {
            metaDesc.setAttribute('content', "Download BioDockify Docking Studio for Windows. A secure, offline desktop molecular docking tool powered by AutoDock Vina. No cloud required.");
        }

        // Fetch latest release
        const fetchLatestRelease = async () => {
            try {
                const response = await fetch('https://api.github.com/repos/tajo9128/Docking-studio/releases/latest');
                if (response.ok) {
                    const data = await response.json();
                    setVersion(data.tag_name);
                    // Find .exe asset
                    const exeAsset = data.assets.find(asset => asset.name.endsWith('.exe'));
                    if (exeAsset) {
                        setDownloadUrl(exeAsset.browser_download_url);
                    }
                }
            } catch (error) {
                console.error("Failed to fetch latest release:", error);
            }
        };

        fetchLatestRelease();
    }, []);

    return (
        <div className="bg-white text-gray-900 font-sans">
            {/* Hero Section */}
            <section className="relative overflow-hidden bg-slate-900 text-white pt-20 pb-32">
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-900 to-slate-900 opacity-90"></div>
                <div className="container mx-auto px-6 relative z-10 text-center">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 border border-indigo-400/30 text-indigo-300 text-sm font-medium mb-6">
                        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
                        {version} Available for Windows
                    </div>
                    <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-300">
                        BioDockify Docking Studio
                    </h1>
                    <p className="text-xl md:text-2xl text-gray-300 mb-10 max-w-3xl mx-auto leading-relaxed">
                        Local, Intelligent Molecular Docking — Built for Real Research.
                        <br className="hidden md:block" />
                        Run secure, reproducible docking workflows entirely on your machine.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <a
                            href={downloadUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-3 bg-indigo-600 hover:bg-indigo-500 text-white px-8 py-4 rounded-xl font-semibold text-lg transition-all transform hover:scale-105 shadow-lg shadow-indigo-500/30 group"
                        >
                            <Download className="w-6 h-6 group-hover:translate-y-1 transition-transform" />
                            Download for Windows
                        </a>
                        <a
                            href="https://github.com/tajo9128/Docking-studio"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-8 py-4 rounded-xl font-medium text-lg transition-all"
                        >
                            View on GitHub
                        </a>
                    </div>
                    <p className="mt-4 text-sm text-gray-400">
                        Requires Windows 11 (x64) & Docker Desktop • Open Source • No Cloud Upload
                    </p>
                </div>
            </section>

            {/* Overview / Value Props */}
            <section className="py-20 bg-gray-50">
                <div className="container mx-auto px-6">
                    <div className="text-center max-w-3xl mx-auto mb-16">
                        <h2 className="text-3xl font-bold mb-4 text-slate-900">Why Use Docking Studio?</h2>
                        <p className="text-gray-600 text-lg">
                            BioDockify Docking Studio combines reliable molecular docking with intelligent failure recovery, clear result interpretation, and a clean desktop workflow.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        <FeatureCard
                            icon={<WifiOff className="w-8 h-8 text-indigo-600" />}
                            title="Offline & Secure"
                            desc="No cloud upload required. No data ever leaves your system. Ideal for confidential projects and IP protection."
                        />
                        <FeatureCard
                            icon={<Activity className="w-8 h-8 text-indigo-600" />}
                            title="Intelligent Recovery"
                            desc="BioDockify AI Agent detects failures (grid errors, interruptions) and automatically applies repair strategies to resume jobs."
                        />
                        <FeatureCard
                            icon={<Cpu className="w-8 h-8 text-indigo-600" />}
                            title="Reproducible Science"
                            desc="Docker-based execution ensures consistent results across different machines. No dependency conflicts."
                        />
                    </div>
                </div>
            </section>

            {/* Detailed Features Grid */}
            <section className="py-20">
                <div className="container mx-auto px-6">
                    <div className="grid md:grid-cols-2 gap-16 items-center">
                        <div>
                            <h2 className="text-3xl font-bold mb-6 text-slate-900">Production-Grade Capabilities</h2>
                            <ul className="space-y-6">
                                <DetailItem
                                    title="Reliable Docking Engine"
                                    desc="Powered by AutoDock Vina. Supports PDB, PDBQT, SDF, MOL2 formats with configurable exhaustiveness."
                                />
                                <DetailItem
                                    title="Fault-Tolerant Workflows"
                                    desc="Detects invalid inputs or corrupted outputs and auto-recovers without restarting the entire job."
                                />
                                <DetailItem
                                    title="Scientific Interaction Analysis"
                                    desc="Clear visualization of H-bonds, hydrophobic contacts, and binding affinity confidence scores."
                                />
                                <DetailItem
                                    title="Zero Configuration"
                                    desc="No Python or development tools needed. Just install Docker Desktop and run the installer."
                                />
                            </ul>
                        </div>
                        <div className="bg-slate-100 rounded-2xl p-8 border border-slate-200">
                            <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
                                <div className="flex items-center gap-3 mb-4 border-b pb-4">
                                    <Terminal className="w-5 h-5 text-gray-500" />
                                    <span className="font-mono text-sm text-gray-600">BioDockify Recovery Agent</span>
                                </div>
                                <div className="space-y-3 font-mono text-sm">
                                    <div className="text-red-500">⚠ Error: Grid box too small for ligand conformer.</div>
                                    <div className="text-indigo-600">➜ Detecting failure type... MATCH [GridError]</div>
                                    <div className="text-indigo-600">➜ Applying Strategy: [ExpandBox +5A]</div>
                                    <div className="text-green-600">✓ Restarting Vina... Success!</div>
                                </div>
                            </div>
                            <div className="text-center text-sm text-gray-500">
                                Real-time failure recovery in action
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* System Requirements */}
            <section className="py-20 bg-slate-50">
                <div className="container mx-auto px-6 max-w-4xl">
                    <h2 className="text-3xl font-bold mb-10 text-center text-slate-900">System Requirements</h2>
                    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
                        <table className="w-full text-left">
                            <tbody className="divide-y divide-gray-100">
                                <ReqRow label="Operating System" value="Windows 11 (x64) recommended" />
                                <ReqRow label="Docker" value="Docker Desktop v4.15 or later" />
                                <ReqRow label="RAM" value="4 GB Minimum (8 GB Recommended)" />
                                <ReqRow label="Disk Space" value="~1 GB free space" />
                                <ReqRow label="Processor" value="Intel/AMD x64 (Apple Silicon coming soon)" />
                            </tbody>
                        </table>
                    </div>
                    <p className="mt-6 text-center text-gray-500 italic">
                        Note: No manual Python or environment setup required. The installer handles dependencies.
                    </p>
                </div>
            </section>

            {/* Usage / CTA */}
            <section className="py-20 text-center">
                <div className="container mx-auto px-6">
                    <ShieldCheck className="w-16 h-16 text-indigo-600 mx-auto mb-6" />
                    <h2 className="text-3xl font-bold mb-6 text-slate-900">Ready to Start?</h2>
                    <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto">
                        BioDockify Docking Studio is open-source and free for academic and commercial use.
                    </p>
                    <a
                        href={downloadUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-3 bg-indigo-600 hover:bg-indigo-500 text-white px-10 py-5 rounded-xl font-bold text-xl shadow-xl transition-transform transform hover:scale-105"
                    >
                        <Download className="w-6 h-6" />
                        Download Installer
                    </a>
                </div>
            </section>
        </div>
    );
};

// Sub-components for cleaner code
const FeatureCard = ({ icon, title, desc }) => (
    <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
        <div className="mb-6 bg-indigo-50 w-16 h-16 rounded-xl flex items-center justify-center">
            {icon}
        </div>
        <h3 className="text-xl font-bold mb-3 text-slate-900">{title}</h3>
        <p className="text-gray-600 leading-relaxed">{desc}</p>
    </div>
);

const DetailItem = ({ title, desc }) => (
    <li className="flex gap-4">
        <div className="mt-1">
            <CheckCircle className="w-6 h-6 text-green-500" />
        </div>
        <div>
            <h4 className="font-bold text-lg text-slate-900">{title}</h4>
            <p className="text-gray-600">{desc}</p>
        </div>
    </li>
);

const ReqRow = ({ label, value }) => (
    <tr className="hover:bg-gray-50/50">
        <td className="p-6 font-semibold text-gray-700 w-1/3">{label}</td>
        <td className="p-6 text-gray-600">{value}</td>
    </tr>
);

export default DockingStudioPage;
