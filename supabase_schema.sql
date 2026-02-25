-- ============================================================
-- Supabase / PostgreSQL schema for Campaign Performance Predictor
-- Run this in the Supabase SQL Editor
-- ============================================================

create extension if not exists "uuid-ossp";

-- Prediction history table
create table if not exists predictions (
    id                       uuid primary key default uuid_generate_v4(),
    caption                  text            not null,
    content                  text            not null default '',
    platform                 varchar(50)     not null,
    post_date                date            not null,
    post_time                time            not null,
    followers                integer         not null default 0,
    ad_boost                 boolean         not null default false,
    pred_likes               numeric(12, 2)  not null default 0,
    pred_comments            numeric(12, 2)  not null default 0,
    pred_shares              numeric(12, 2)  not null default 0,
    pred_clicks              numeric(12, 2)  not null default 0,
    pred_timing_quality_score numeric(10, 4) not null default 0,
    created_at               timestamptz     not null default now()
);

-- Index for recent-first queries
create index if not exists predictions_created_at_idx
    on predictions (created_at desc);

-- Enable Row Level Security (open read/write for service key)
alter table predictions enable row level security;

-- Policy: allow all operations from authenticated / service role
create policy "Allow all for service role"
    on predictions
    for all
    using (true)
    with check (true);
