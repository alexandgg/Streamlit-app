\ Model RAP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.0965368699462694 assign[SPY] + 0.0969713729065524 assign[IVV]
   + 0.0887216342078145 assign[VTI] + 0.097126489832086 assign[VOO]
   + 0.1940998225645225 assign[QQQ] + 0.013558654892424 assign[VEA]
   + 0.0153949752371305 assign[IEFA] + 0.0136358175889657 assign[VWO]
   + 0.142888641571316 assign[VUG] + 0.0167286860811857 assign[IEMG]
Subject To
 assets[SPY]: assign[SPY] = 1
 assets[IVV]: assign[IVV] = 1
 assets[VTI]: assign[VTI] = 1
 assets[VOO]: assign[VOO] = 1
 assets[QQQ]: assign[QQQ] = 1
 assets[VEA]: assign[VEA] = 1
 assets[IEFA]: assign[IEFA] = 1
 assets[VWO]: assign[VWO] = 1
 assets[VUG]: assign[VUG] = 1
 assets[IEMG]: assign[IEMG] = 1
Bounds
End
