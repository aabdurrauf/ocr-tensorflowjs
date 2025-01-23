// config-overrides.js

module.exports = function override(config, env) {
    // Modify the Webpack configuration to include the necessary polyfills
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: require.resolve('browserify-fs'),
      crypto: require.resolve('crypto-browserify'),
      stream: require.resolve('stream-browserify'), 
    };
  
    return config;
  };
  