import time
import json
from collections import defaultdict

class Profiler:
    # Structure: stats[category][operation_name] = total_time_seconds
    stats = defaultdict(lambda: defaultdict(float))
    calls = defaultdict(lambda: defaultdict(int))
    current_category = "General"
    enabled = True

    @classmethod
    def profile(cls, name):
        """Decorator to measure execution time of a specific hardware block."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not cls.enabled:
                    return func(*args, **kwargs)
                
                t0 = time.perf_counter()
                result = func(*args, **kwargs)
                t1 = time.perf_counter()
                
                elapsed = t1 - t0
                cls.stats[cls.current_category][name] += elapsed
                cls.calls[cls.current_category][name] += 1
                return result
            return wrapper
        return decorator

    @classmethod
    def generate_html_dashboard(cls, filename="benchmark_dashboard.html"):
        """Exports the aggregated data into a standalone interactive HTML dashboard."""
        if not cls.stats:
            return

        # Prepare data for Chart.js
        categories = list(cls.stats.keys())
        operations = set()
        for cat_data in cls.stats.values():
            operations.update(cat_data.keys())
        operations = list(operations)

        # Build datasets for a stacked bar chart
        datasets = []
        colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6"]
        
        for idx, op in enumerate(operations):
            data = []
            for cat in categories:
                # Convert to milliseconds for readability
                val = cls.stats[cat].get(op, 0.0) * 1000 
                
                # Average it out by the number of tests in that category
                # (Assuming ~10 tests per category for rough averaging, or use raw absolute time)
                data.append(round(val, 3))
                
            datasets.append({
                "label": op,
                "data": data,
                "backgroundColor": colors[idx % len(colors)]
            })

        chart_data = {
            "labels": categories,
            "datasets": datasets
        }

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kyber Hardware Pipeline Benchmarks</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: #1e293b; padding: 2rem; border-radius: 12px; }}
                h1 {{ text-align: center; color: #38bdf8; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Kyber KEM Execution Profiler</h1>
                <p style="text-align: center;">Absolute Time (ms) per Category Breakdown</p>
                <canvas id="benchmarkChart"></canvas>
            </div>
            <script>
                const ctx = document.getElementById('benchmarkChart').getContext('2d');
                new Chart(ctx, {{
                    type: 'bar',
                    data: {json.dumps(chart_data)},
                    options: {{
                        responsive: true,
                        scales: {{
                            x: {{ stacked: true }},
                            y: {{ stacked: true, title: {{ display: true, text: 'Total Time (ms)' }} }}
                        }},
                        plugins: {{
                            tooltip: {{ mode: 'index', intersect: false }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        with open(filename, "w") as f:
            f.write(html_content)
        print(f"\n[+] Interactive benchmark dashboard generated: {filename}")