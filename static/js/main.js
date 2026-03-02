// SmartJob AI – Main JavaScript

// Auto-dismiss flash messages
document.querySelectorAll('.alert').forEach(el => {
  setTimeout(() => {
    const bsAlert = bootstrap.Alert.getOrCreateInstance(el);
    bsAlert?.close();
  }, 5000);
});

// Animate skill match progress bars on page load
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.skill-bar-fill').forEach(bar => {
    const target = bar.style.width;
    bar.style.width = '0%';
    requestAnimationFrame(() => {
      setTimeout(() => { bar.style.width = target; }, 100);
    });
  });
});
