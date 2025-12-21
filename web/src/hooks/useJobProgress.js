import { useState, useEffect } from 'react';

const SOCKET_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

export function useJobProgress(jobId) {
    const [progress, setProgress] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState(null);

    useEffect(() => {
        if (!jobId) return;

        console.log(`🔌 Connecting to WS for Job ${jobId}...`);
        const ws = new WebSocket(`${SOCKET_URL}/jobs/${jobId}`);

        ws.onopen = () => {
            console.log('✅ WS Connected');
            setIsConnected(true);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setLastMessage(data);
                if (data.progress !== undefined) {
                    setProgress(data.progress);
                }
            } catch (err) {
                console.error('WS Parse Error:', err);
            }
        };

        ws.onclose = () => {
            console.log('❌ WS Disconnected');
            setIsConnected(false);
        };

        return () => {
            ws.close();
        };
    }, [jobId]);

    return { progress, isConnected, lastMessage };
}
