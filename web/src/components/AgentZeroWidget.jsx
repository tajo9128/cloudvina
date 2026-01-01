import React, { useState, useRef, useEffect } from 'react';
import { supabase } from '../supabaseClient';
import ReactMarkdown from 'react-markdown';

export default function AgentZeroWidget() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'assistant', text: "Hello! I am **Agent Zero**. I can help you interpret docking results or suggest next steps. How can I help?" }
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isOpen]);

    // NEW: Listen for external triggers (e.g., from JobResultsPage)
    useEffect(() => {
        const handleTrigger = (event) => {
            const { prompt, autoSend } = event.detail;
            setIsOpen(true);
            if (prompt) {
                setInput(prompt);
                if (autoSend) {
                    // Slight delay to ensure state updates
                    setTimeout(() => {
                        handleSubmit(null, prompt);
                    }, 100);
                }
            }
        };

        window.addEventListener('agent-zero-trigger', handleTrigger);
        return () => window.removeEventListener('agent-zero-trigger', handleTrigger);
    }, []);

    const handleSubmit = async (e, forcedInput = null) => {
        if (e) e.preventDefault();
        const textToSend = forcedInput || input;
        if (!textToSend.trim()) return;

        const userMsg = { role: 'user', text: textToSend };
        setMessages(prev => [...prev, userMsg]);
        setInput("");
        setIsLoading(true);

        try {
            const { data: { session } } = await supabase.auth.getSession();
            const token = session?.access_token;

            // 1. Detect Context (Job Results)
            let contextData = { page: window.location.pathname };
            let contextType = 'general';

            // Check if we are on a job page
            const jobMatch = window.location.pathname.match(/\/dock\/([a-zA-Z0-9-]+)/);
            if (jobMatch && jobMatch[1] && jobMatch[1] !== 'new' && jobMatch[1] !== 'batch') {
                const jobId = jobMatch[1];
                try {
                    // Fetch minimal job details for context
                    const jobRes = await fetch(`${import.meta.env.VITE_API_URL}/jobs/${jobId}`, {
                        headers: { 'Authorization': `Bearer ${token}` }
                    });
                    if (jobRes.ok) {
                        const jobJson = await jobRes.json();
                        contextData = {
                            job_id: jobId,
                            ligand: jobJson.ligand_filename,
                            receptor: jobJson.receptor_filename,
                            affinity: jobJson.binding_affinity,
                            status: jobJson.status
                        };
                        // If user asks "explain", switch mode
                        if (input.toLowerCase().includes('explain') || input.toLowerCase().includes('interpret')) {
                            contextType = 'result_explanation';
                        } else if (input.toLowerCase().includes('next') || input.toLowerCase().includes('now what')) {
                            contextType = 'next_steps';
                        }
                    }
                } catch (err) {
                    console.warn("Agent Zero: Could not fetch job context", err);
                }
            }

            const response = await fetch(`${import.meta.env.VITE_API_URL}/agent/consult`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    query: textToSend,
                    context_type: contextType,
                    data: contextData
                })
            });

            if (!response.ok) throw new Error("Agent busy");

            const data = await response.json();
            const replyText = data.analysis || data.suggestion || JSON.stringify(data);

            setMessages(prev => [...prev, { role: 'assistant', text: replyText }]);
        } catch (error) {
            setMessages(prev => [...prev, { role: 'assistant', text: "⚠️ I'm having trouble connecting to my brain. Please check if the backend is running." }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
            {/* Chat Window */}
            {isOpen && (
                <div className="mb-4 w-80 md:w-96 bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden flex flex-col h-[500px] animate-fade-in-up">
                    {/* Header */}
                    <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-4 flex justify-between items-center text-white">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
                            <h3 className="font-bold text-sm">Agent Zero (v7.0)</h3>
                        </div>
                        <button onClick={() => setIsOpen(false)} className="hover:bg-white/20 rounded p-1">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[85%] rounded-lg p-3 text-sm ${msg.role === 'user'
                                    ? 'bg-indigo-600 text-white rounded-br-none'
                                    : 'bg-white border border-slate-200 text-slate-700 shadow-sm rounded-bl-none'
                                    }`}>
                                    <ReactMarkdown className="prose prose-sm max-w-none dark:prose-invert">
                                        {msg.text}
                                    </ReactMarkdown>
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="bg-white border border-slate-200 rounded-lg p-3 rounded-bl-none flex gap-1">
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-75"></div>
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-150"></div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <form onSubmit={handleSubmit} className="p-3 bg-white border-t border-slate-100 flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask Agent Zero..."
                            className="flex-1 text-sm border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        />
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg px-3 py-2 disabled:opacity-50 transition-colors"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
                        </button>
                    </form>
                </div>
            )}

            {/* Toggle Button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`flex items-center justify-center w-14 h-14 rounded-full shadow-lg transition-all transform hover:scale-110 ${isOpen ? 'bg-slate-700 text-slate-300' : 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                    }`}
            >
                {isOpen ? (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                ) : (
                    <div className="relative">
                        <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
                        <span className="absolute -top-1 -right-1 flex h-3 w-3">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                        </span>
                    </div>
                )}
            </button>
        </div>
    );
}
