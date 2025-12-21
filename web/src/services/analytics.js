import mixpanel from 'mixpanel-browser';

const MIXPANEL_TOKEN = import.meta.env.VITE_MIXPANEL_TOKEN;

export const initAnalytics = () => {
    if (MIXPANEL_TOKEN) {
        try {
            mixpanel.init(MIXPANEL_TOKEN, {
                debug: import.meta.env.DEV,
                track_pageview: true,
                persistence: 'localStorage',
                ignore_dnt: true
            });
            console.log("Analytics initialized");
        } catch (error) {
            console.warn("Mixpanel initialization failed:", error);
        }
    }
};

export const trackEvent = (event, properties = {}) => {
    if (!MIXPANEL_TOKEN) return;
    try {
        mixpanel.track(event, properties);
    } catch (error) {
        console.warn("Analytics track error:", error);
    }
};

export const identifyUser = (userId, traits = {}) => {
    if (!MIXPANEL_TOKEN) return;
    try {
        mixpanel.identify(userId);
        if (Object.keys(traits).length > 0) {
            mixpanel.people.set(traits);
        }
    } catch (error) {
        console.warn("Analytics identify error:", error);
    }
};

export const pageView = (name, properties = {}) => {
    if (!MIXPANEL_TOKEN) return;
    try {
        mixpanel.track_pageview(properties);
    } catch (e) { }
};

export default { initAnalytics, trackEvent, identifyUser, pageView };
