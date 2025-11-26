/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Deep Blue Palette (Antler Home3D inspired)
                'deep-navy': {
                    50: '#e6eef7',
                    100: '#ccddef',
                    200: '#99bbe0',
                    300: '#6699d0',
                    400: '#3377c1',
                    500: '#0055b1',
                    600: '#00448e',
                    700: '#00336a',
                    800: '#002247',
                    900: '#0a1e3e',
                    950: '#050f1f',
                },
                'royal-blue': {
                    50: '#eff6ff',
                    100: '#dbeafe',
                    200: '#bfdbfe',
                    300: '#93c5fd',
                    400: '#60a5fa',
                    500: '#3b82f6',
                    600: '#2563eb',
                    700: '#1e3a8a',
                    800: '#1e3a8a',
                    900: '#1e40af',
                },
                'tech-cyan': {
                    50: '#ecfeff',
                    100: '#cffafe',
                    200: '#a5f3fc',
                    300: '#67e8f9',
                    400: '#22d3ee',
                    500: '#06b6d4',
                    600: '#0891b2',
                    700: '#0e7490',
                    800: '#155e75',
                    900: '#164e63',
                },
            },
            backgroundImage: {
                'blue-gradient': 'linear-gradient(to bottom right, #1e3a8a, #0a1e3e)',
                'blue-gradient-radial': 'radial-gradient(circle, #3b82f6, #1e3a8a)',
                'tech-mesh': 'linear-gradient(135deg, #0a1e3e 0%, #1e3a8a 25%, #3b82f6 50%, #1e3a8a 75%, #0a1e3e 100%)',
            },
            boxShadow: {
                'blue-glow': '0 0 20px rgba(59, 130, 246, 0.5)',
                'blue-glow-lg': '0 0 40px rgba(59, 130, 246, 0.6)',
                'deep-blue': '0 10px 40px rgba(10, 30, 62, 0.3)',
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'float': 'float 6s ease-in-out infinite',
            },
            keyframes: {
                float: {
                    '0%, 100%': { transform: 'translateY(0px)' },
                    '50%': { transform: 'translateY(-20px)' },
                },
            },
        },
    },
    plugins: [],
}
