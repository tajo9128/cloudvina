import mixpanel from 'mixpanel-browser';

// Initialize Mixpanel
export const initAnalytics = () => {
    try {
        // Replace with actual token from env or config
        // For now, we wrap in try-catch to prevent crash if not installed/configured
        if (import.meta.env.VITE_MIXPANEL_TOKEN) {
            mixpanel.init(import.meta.env.VITE_MIXPANEL_TOKEN, {
                debug: import.meta.env.DEV,
                track_pageview: true,
                persistence: 'localStorage'
            });
        } else {
            console.warn('Mixpanel Token not found (VITE_MIXPANEL_TOKEN)');
        }
    } catch (error) {
        console.error('Analytics Init Failed (Safe Fail):', error);
    }
};

// Track Event (Safe)
export const trackEvent = (eventName, properties = {}) => {
    try {
        if (import.meta.env.VITE_MIXPANEL_TOKEN) {
            mixpanel.track(eventName, {
                ...properties,
                timestamp: new Date().toISOString(),
                // detailed context
            });
        } else {
            // Dev / Fallback Logging
            if (import.meta.env.DEV) {
                console.log(`[Analytics] ${eventName}`, properties);
            }
        }
    } catch (error) {
        // Silent fail to protect user experience
        console.warn('Analytics Track Failed (Safe Fail):', error);
    }
};

// Identify User (Safe)
export const identifyUser = (userId, userTraits = {}) => {
    try {
        if (import.meta.env.VITE_MIXPANEL_TOKEN) {
            mixpanel.identify(userId);
            mixpanel.people.set(userTraits);
        }
    } catch (error) {
        console.warn('Analytics Identify Failed (Safe Fail):', error);
    }
};
