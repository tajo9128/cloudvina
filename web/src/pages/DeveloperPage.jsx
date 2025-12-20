import React from 'react';
import { Terminal, Download, BookOpen, Code, ChevronRight, Boxes } from 'lucide-react';
import { API_URL } from '../config';

export default function DeveloperPage() {
    const sdkUrl = `${API_URL}/static/sdk/biodockify.py`; // We need to expose this static file or just serve it as blob

    const handleDownloadSDK = () => {
        // Simple client-side download for now
        // In a real app, we'd fetch the file from the API or serve it via static folder
        const sdkContent = `import requests\nimport os\n\nclass BioDockify:...\nprint('Please download full SDK from GitHub or API')`;

        // Since we created the file in 'sdk/biodockify.py' which is outside 'web', 
        // we can't link to it easily without an API endpoint serving it.
        // Quick Hack: Alert user or provide GitHub link. 
        // Better: We will create a blob here for the User to feel the magic.

        alert("In this demo, please check the 'sdk/' folder in the repository for 'biodockify.py'.");
    };

    return (
        <div className="min-h-screen bg-slate-50">
            {/* HERO */}
            <div className="bg-slate-900 text-white pt-20 pb-16 px-6">
                <div className="container mx-auto max-w-5xl">
                    <div className="flex flex-col md:flex-row items-center gap-12">
                        <div className="flex-1 space-y-6">
                            <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-500/20 text-indigo-300 rounded-full text-xs font-bold border border-indigo-500/30">
                                <Terminal className="w-3 h-3" /> v6.0.0 Now Available
                            </div>
                            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
                                Build on <span className="text-indigo-400">BioDockify</span>
                            </h1>
                            <p className="text-lg text-slate-400 leading-relaxed">
                                Integrate molecular docking into your drug discovery pipelines with our powerful Python SDK.
                                Automate batch jobs, retrieve results, and scale effortlessly.
                            </p>
                            <div className="flex gap-4 pt-2">
                                <button className="px-6 py-3 bg-indigo-600 text-white font-bold rounded-lg hover:bg-indigo-700 transition flex items-center gap-2">
                                    <BookOpen className="w-4 h-4" /> Read Docs
                                </button>
                                <button className="px-6 py-3 bg-slate-800 text-white font-bold rounded-lg hover:bg-slate-700 transition flex items-center gap-2 border border-slate-700">
                                    <Download className="w-4 h-4" /> Download SDK
                                </button>
                            </div>
                        </div>
                        <div className="flex-1 w-full relative">
                            {/* Code Window */}
                            <div className="bg-[#1e1e1e] rounded-xl shadow-2xl border border-slate-700 overflow-hidden text-sm font-mono">
                                <div className="flex items-center justify-between px-4 py-3 bg-[#252526] border-b border-slate-700">
                                    <div className="flex gap-2">
                                        <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                        <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                                        <div className="w-3 h-3 rounded-full bg-green-500"></div>
                                    </div>
                                    <div className="text-slate-500 text-xs">main.py</div>
                                </div>
                                <div className="p-6 text-slate-300 overflow-x-auto">
                                    <pre>{`from biodockify import BioDockify

# 1. Initialize Client
client = BioDockify(
    email="dr.strange@research.lab", 
    password="***"
)

# 2. Submit Batch Job
job_id = client.submit_job(
    receptor_path="./targets/cov2_spike.pdb",
    ligand_path="./leads/compound_x.sdf",
    config={"exhaustiveness": 16}
)

# 3. Wait for Results
results = client.wait_for_completion(job_id)
print(f"Binding Affinity: {results['binding_affinity']}")
`}</pre>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* FEATURES */}
            <div className="py-20 px-6 container mx-auto max-w-5xl">
                <div className="grid md:grid-cols-3 gap-8">
                    <div className="bg-white p-6 rounded-xl border border-slate-200 hover:shadow-lg transition">
                        <div className="w-12 h-12 bg-indigo-100 rounded-xl flex items-center justify-center mb-4 text-indigo-600">
                            <Boxes className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900 mb-2">Rest API</h3>
                        <p className="text-slate-600 text-sm leading-relaxed mb-4">
                            Full access to all platform features via standard REST endpoints. Secured with JWT.
                        </p>
                        <a href={`${API_URL}/docs`} target="_blank" className="text-indigo-600 font-bold text-sm flex items-center gap-2 hover:underline">
                            Explore Swagger UI <ChevronRight className="w-4 h-4" />
                        </a>
                    </div>

                    <div className="bg-white p-6 rounded-xl border border-slate-200 hover:shadow-lg transition">
                        <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center mb-4 text-emerald-600">
                            <Code className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900 mb-2">Python SDK</h3>
                        <p className="text-slate-600 text-sm leading-relaxed mb-4">
                            Type-safe wrapper for rapid development. Handles authentication, S3 uploads, and polling.
                        </p>
                        <a href="#" className="text-emerald-600 font-bold text-sm flex items-center gap-2 hover:underline">
                            View Source <ChevronRight className="w-4 h-4" />
                        </a>
                    </div>

                    <div className="bg-white p-6 rounded-xl border border-slate-200 hover:shadow-lg transition">
                        <div className="w-12 h-12 bg-amber-100 rounded-xl flex items-center justify-center mb-4 text-amber-600">
                            <Terminal className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900 mb-2">CLI Tool (Beta)</h3>
                        <p className="text-slate-600 text-sm leading-relaxed mb-4">
                            Run jobs directly from your terminal. Great for integration with HPC clusters.
                        </p>
                        <a href="#" className="text-amber-600 font-bold text-sm flex items-center gap-2 hover:underline">
                            Learn More <ChevronRight className="w-4 h-4" />
                        </a>
                    </div>
                </div>
            </div>
        </div>
    );
}
