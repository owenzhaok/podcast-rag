import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import SearchBar from '../src/components/SearchBar';

describe('SearchBar', () => {
  it('renders input and button', () => {
    render(<SearchBar onSearch={vi.fn()} loading={false} />);
    expect(screen.getByPlaceholderText('Search podcasts...')).toBeDefined();
    expect(screen.getByRole('button', { name: 'Search' })).toBeDefined();
  });

  it('calls onSearch on enter', async () => {
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} loading={false} />);
    const input = screen.getByPlaceholderText('Search podcasts...');
    await userEvent.type(input, 'machine learning{Enter}');
    expect(onSearch).toHaveBeenCalledWith('machine learning', 2);
  });

  it('calls onSearch on button click', async () => {
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} loading={false} />);
    await userEvent.type(screen.getByPlaceholderText('Search podcasts...'), 'AI');
    await userEvent.click(screen.getByRole('button', { name: 'Search' }));
    expect(onSearch).toHaveBeenCalledWith('AI', 2);
  });

  it('disables input while loading', () => {
    render(<SearchBar onSearch={vi.fn()} loading={true} />);
    expect(screen.getByPlaceholderText('Search podcasts...')).toHaveProperty('disabled', true);
  });

  it('does not submit empty query', async () => {
    const onSearch = vi.fn();
    render(<SearchBar onSearch={onSearch} loading={false} />);
    await userEvent.keyboard('{Enter}');
    expect(onSearch).not.toHaveBeenCalled();
  });
});
