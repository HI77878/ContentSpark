-- PostgreSQL Schema for TikTok Video Analyzer
-- Stores analysis results, metadata, and performance metrics

-- Create database (run as postgres user)
-- CREATE DATABASE tiktok_analyzer;
-- \c tiktok_analyzer;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For text search

-- Main video analyses table
CREATE TABLE IF NOT EXISTS video_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tiktok_url VARCHAR(255) UNIQUE NOT NULL,
    video_id VARCHAR(50) NOT NULL,
    creator_username VARCHAR(100),
    creator_id VARCHAR(50),
    
    -- Video metadata
    title TEXT,
    description TEXT,
    hashtags TEXT[], -- Array of hashtags
    duration_seconds FLOAT NOT NULL,
    view_count BIGINT,
    like_count BIGINT,
    comment_count BIGINT,
    share_count BIGINT,
    upload_date DATE,
    
    -- Processing metadata
    processing_time_seconds FLOAT NOT NULL,
    realtime_factor FLOAT GENERATED ALWAYS AS (processing_time_seconds / NULLIF(duration_seconds, 0)) STORED,
    analyzer_version VARCHAR(20),
    successful_analyzers INTEGER,
    total_analyzers INTEGER,
    reconstruction_score FLOAT,
    
    -- Analysis results (JSONB for flexibility)
    analyzer_results JSONB NOT NULL,
    
    -- File paths
    video_file_path TEXT,
    result_file_path TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    status VARCHAR(20) DEFAULT 'completed',
    error_message TEXT
);

-- Indexes for fast queries
CREATE INDEX idx_creator_username ON video_analyses(creator_username);
CREATE INDEX idx_video_id ON video_analyses(video_id);
CREATE INDEX idx_created_at ON video_analyses(created_at DESC);
CREATE INDEX idx_processing_time ON video_analyses(processing_time_seconds);
CREATE INDEX idx_realtime_factor ON video_analyses(realtime_factor);
CREATE INDEX idx_hashtags ON video_analyses USING GIN(hashtags);
CREATE INDEX idx_analyzer_results ON video_analyses USING GIN(analyzer_results);

-- Full text search on description
CREATE INDEX idx_description_search ON video_analyses USING GIN(to_tsvector('english', description));

-- Analyzer performance tracking
CREATE TABLE IF NOT EXISTS analyzer_performance (
    id SERIAL PRIMARY KEY,
    analyzer_name VARCHAR(50) NOT NULL,
    video_analysis_id UUID REFERENCES video_analyses(id) ON DELETE CASCADE,
    
    -- Performance metrics
    processing_time_seconds FLOAT NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    
    -- Quality metrics
    duplicate_rate FLOAT,
    data_completeness FLOAT,
    confidence_score FLOAT,
    
    -- Resource usage
    peak_gpu_memory_mb FLOAT,
    peak_cpu_percent FLOAT,
    
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analyzer_name ON analyzer_performance(analyzer_name);
CREATE INDEX idx_analyzer_timestamp ON analyzer_performance(timestamp DESC);
CREATE INDEX idx_analyzer_success ON analyzer_performance(success);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- GPU metrics
    gpu_usage_percent FLOAT,
    gpu_memory_used_gb FLOAT,
    gpu_memory_total_gb FLOAT,
    gpu_temperature FLOAT,
    gpu_power_watts FLOAT,
    
    -- CPU/System metrics
    cpu_usage_percent FLOAT,
    ram_usage_percent FLOAT,
    ram_used_gb FLOAT,
    ram_total_gb FLOAT,
    disk_usage_percent FLOAT,
    
    -- API metrics
    active_connections INTEGER,
    queue_size INTEGER,
    processing_jobs INTEGER
);

CREATE INDEX idx_metrics_timestamp ON system_metrics(timestamp DESC);

-- Processing queue table
CREATE TABLE IF NOT EXISTS processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tiktok_url VARCHAR(255) NOT NULL,
    priority INTEGER DEFAULT 3, -- 1=urgent, 2=high, 3=normal, 4=low
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    
    -- Job metadata
    creator_username VARCHAR(100),
    requested_analyzers TEXT[],
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Results
    video_analysis_id UUID REFERENCES video_analyses(id),
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    CONSTRAINT unique_pending_url UNIQUE (tiktok_url, status)
);

CREATE INDEX idx_queue_status ON processing_queue(status);
CREATE INDEX idx_queue_priority ON processing_queue(priority, created_at);

-- Daily statistics view
CREATE OR REPLACE VIEW daily_statistics AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as videos_processed,
    COUNT(DISTINCT creator_username) as unique_creators,
    AVG(duration_seconds) as avg_video_duration,
    AVG(processing_time_seconds) as avg_processing_time,
    AVG(realtime_factor) as avg_realtime_factor,
    AVG(reconstruction_score) as avg_reconstruction_score,
    SUM(duration_seconds) as total_video_seconds,
    SUM(processing_time_seconds) as total_processing_seconds
FROM video_analyses
WHERE status = 'completed'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Analyzer statistics view
CREATE OR REPLACE VIEW analyzer_statistics AS
SELECT 
    analyzer_name,
    COUNT(*) as total_runs,
    COUNT(*) FILTER (WHERE success = true) as successful_runs,
    COUNT(*) FILTER (WHERE success = false) as failed_runs,
    AVG(processing_time_seconds) as avg_processing_time,
    MIN(processing_time_seconds) as min_processing_time,
    MAX(processing_time_seconds) as max_processing_time,
    AVG(duplicate_rate) as avg_duplicate_rate,
    AVG(data_completeness) as avg_completeness,
    AVG(confidence_score) as avg_confidence
FROM analyzer_performance
WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY analyzer_name
ORDER BY total_runs DESC;

-- Function to get video by URL or ID
CREATE OR REPLACE FUNCTION get_video_analysis(url_or_id TEXT)
RETURNS TABLE(LIKE video_analyses)
AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM video_analyses
    WHERE tiktok_url = url_or_id 
       OR video_id = url_or_id
       OR id::text = url_or_id
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to search videos by creator
CREATE OR REPLACE FUNCTION search_by_creator(username TEXT, limit_count INTEGER DEFAULT 100)
RETURNS TABLE(LIKE video_analyses)
AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM video_analyses
    WHERE creator_username ILIKE '%' || username || '%'
    ORDER BY created_at DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to search videos by hashtag
CREATE OR REPLACE FUNCTION search_by_hashtag(hashtag TEXT, limit_count INTEGER DEFAULT 100)
RETURNS TABLE(LIKE video_analyses)
AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM video_analyses
    WHERE hashtag = ANY(hashtags)
    ORDER BY created_at DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update modified timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_video_analyses_modtime
    BEFORE UPDATE ON video_analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Materialized view for fast creator statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS creator_statistics AS
SELECT 
    creator_username,
    creator_id,
    COUNT(*) as video_count,
    AVG(duration_seconds) as avg_duration,
    SUM(view_count) as total_views,
    SUM(like_count) as total_likes,
    AVG(reconstruction_score) as avg_reconstruction_score,
    MIN(created_at) as first_analyzed,
    MAX(created_at) as last_analyzed
FROM video_analyses
WHERE creator_username IS NOT NULL
GROUP BY creator_username, creator_id;

CREATE UNIQUE INDEX idx_creator_stats_username ON creator_statistics(creator_username);

-- Refresh materialized view function
CREATE OR REPLACE FUNCTION refresh_creator_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY creator_statistics;
END;
$$ LANGUAGE plpgsql;

-- Sample queries:
/*
-- Get recent analyses
SELECT * FROM video_analyses 
ORDER BY created_at DESC 
LIMIT 10;

-- Get performance metrics for today
SELECT * FROM daily_statistics 
WHERE date = CURRENT_DATE;

-- Find slow analyses
SELECT tiktok_url, duration_seconds, processing_time_seconds, realtime_factor
FROM video_analyses
WHERE realtime_factor > 3
ORDER BY realtime_factor DESC;

-- Get analyzer performance
SELECT * FROM analyzer_statistics;

-- Search by creator
SELECT * FROM search_by_creator('username');

-- Get videos with specific hashtag
SELECT * FROM search_by_hashtag('morningroutine');
*/