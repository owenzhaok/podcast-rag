import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ClipCard from '../src/components/ClipCard';
import { ClipResult } from '../src/types';

function makeClip(overrides: Partial<ClipResult> = {}): ClipResult {
  return {
    podcast_id: 'show_abc',
    episode_id: 'ep001',
    show_name: 'Test Show',
    episode_name: 'Test Episode',
    clip_start_ms: 60000,
    clip_end_ms: 180000,
    score: 5.0,
    highlight: 'This is a <em>test</em> highlight',
    word_timestamps: [],
    speakers: [1],
    audio_link: null,
    ...overrides,
  };
}

describe('ClipCard', () => {
  it('renders show and episode name', () => {
    render(<ClipCard clip={makeClip()} />);
    expect(screen.getByText('Test Show')).toBeDefined();
    expect(screen.getByText('Test Episode')).toBeDefined();
  });

  it('renders timestamp range', () => {
    render(<ClipCard clip={makeClip()} />);
    expect(screen.getByText(/1:00/)).toBeDefined();
    expect(screen.getByText(/3:00/)).toBeDefined();
  });

  it('renders highlighted excerpt as HTML', () => {
    const { container } = render(<ClipCard clip={makeClip()} />);
    const em = container.querySelector('em');
    expect(em).not.toBeNull();
    expect(em!.textContent).toBe('test');
  });

  it('hides audio link when null', () => {
    render(<ClipCard clip={makeClip({ audio_link: null })} />);
    expect(screen.queryByText('Listen')).toBeNull();
  });

  it('shows audio link when present', () => {
    render(<ClipCard clip={makeClip({ audio_link: 'https://example.com/audio.mp3' })} />);
    const link = screen.getByText('Listen');
    expect(link).toBeDefined();
    expect(link.getAttribute('href')).toBe('https://example.com/audio.mp3');
  });

  it('shows speaker badge for multiple speakers', () => {
    render(<ClipCard clip={makeClip({ speakers: [1, 2] })} />);
    expect(screen.getByText('2 speakers')).toBeDefined();
  });
});
