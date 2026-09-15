// Initialize Mermaid with custom configuration
document$.subscribe(function() {
  mermaid.initialize({
    startOnLoad: true,
    theme: 'base',
    themeVariables: {
      primaryColor: '#1565c0',
      primaryTextColor: '#ffffff',
      primaryBorderColor: '#0d47a1',
      lineColor: '#42a5f5',
      sectionBkgColor: '#e3f2fd',
      altSectionBkgColor: '#bbdefb',
      gridColor: '#90caf9',
      secondaryColor: '#ff9800',
      tertiaryColor: '#4caf50'
    },
    fontFamily: 'Roboto, sans-serif',
    sequence: {
      diagramMarginX: 50,
      diagramMarginY: 10,
      actorMargin: 50,
      width: 150,
      height: 65,
      boxMargin: 10,
      boxTextMargin: 5,
      noteMargin: 10,
      messageMargin: 35,
      mirrorActors: true,
      bottomMarginAdj: 1,
      useMaxWidth: true
    },
    flowchart: {
      htmlLabels: false,
      curve: 'basis'
    }
  });
});
