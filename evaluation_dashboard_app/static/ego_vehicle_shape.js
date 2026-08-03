(function () {
  const wheelBase = 2.79;
  const frontOverhang = 1.0;
  const rearOverhang = 1.1;
  const wheelTread = 1.64;
  const leftOverhang = 0.128;
  const rightOverhang = 0.128;
  const width = wheelTread + leftOverhang + rightOverhang;
  const rear = -(wheelBase / 2 + rearOverhang);
  const front = wheelBase / 2 + frontOverhang;
  const halfWidth = width / 2;
  const shoulder = halfWidth - .18;
  const z = .02;
  const deckZ = z + .58;
  const beltZ = z + .82;
  const roofZ = z + 1.44;
  const railZ = z + 1.56;
  const rearWheelX = -wheelBase / 2;
  const frontWheelX = wheelBase / 2;
  const wheelY = wheelTread / 2;
  const cabinRear = -1.0;
  const cabinFront = .84;
  const glassRear = -.62;
  const glassFront = .52;
  const hoodRear = .82;
  const trunkFront = -1.12;

  const style = (token, alpha) => ({token, alpha});
  const face = (name, points, fill, stroke, width) => ({name, points, fill, stroke, width});
  const line = (name, points, stroke, width) => ({name, points, stroke, width});

  const faces = [
    face("footprint-shadow", [[rear, -halfWidth, z], [front, -halfWidth, z], [front, halfWidth, z], [rear, halfWidth, z]], style("deep", .14), style("lineStrong", .24), .7),
    face("left-lower-door", [[rear + .12, -halfWidth, z + .04], [front - .16, -halfWidth, z + .04], [front - .34, -shoulder, deckZ], [rear + .2, -shoulder, deckZ]], style("surface", .38), style("lineStrong", .44), .9),
    face("right-lower-door", [[rear + .12, halfWidth, z + .04], [front - .16, halfWidth, z + .04], [front - .34, shoulder, deckZ], [rear + .2, shoulder, deckZ]], style("surface", .54), style("lineStrong", .52), .9),
    face("front-bumper", [[front - .18, -halfWidth, z + .04], [front, -.62, z + .16], [front, .62, z + .16], [front - .18, halfWidth, z + .04], [front - .34, shoulder, deckZ], [front - .34, -shoulder, deckZ]], style("accent", .18), style("accentBright", .68), 1.1),
    face("rear-bumper", [[rear, -.62, z + .14], [rear, .62, z + .14], [rear + .2, shoulder, deckZ], [rear + .2, -shoulder, deckZ]], style("lineStrong", .12), style("lineStrong", .74), 1.1),
    face("upper-body", [[rear + .2, -shoulder, deckZ], [front - .34, -shoulder, deckZ], [front - .34, shoulder, deckZ], [rear + .2, shoulder, deckZ]], style("surface", .78), style("lineStrong", .62), 1.1),
    face("hood", [[hoodRear, -.5, deckZ + .03], [front - .48, -.58, deckZ], [front - .2, 0, deckZ + .08], [front - .48, .58, deckZ], [hoodRear, .5, deckZ + .03]], style("surface", .58), style("accent", .36), .9),
    face("rear-deck", [[rear + .28, -.58, deckZ], [trunkFront, -.5, deckZ + .04], [trunkFront, .5, deckZ + .04], [rear + .28, .58, deckZ]], style("lineStrong", .07), style("lineStrong", .36), .9),
    face("left-glass", [[cabinRear, -.56, beltZ], [cabinFront, -.52, beltZ], [glassFront, -.33, roofZ], [glassRear, -.36, roofZ]], style("deep", .56), style("accent", .58), .9),
    face("right-glass", [[cabinRear, .56, beltZ], [cabinFront, .52, beltZ], [glassFront, .33, roofZ], [glassRear, .36, roofZ]], style("deep", .62), style("accent", .62), .9),
    face("windshield", [[cabinFront, -.52, beltZ], [cabinFront, .52, beltZ], [glassFront, .33, roofZ], [glassFront, -.33, roofZ]], style("accentBright", .18), style("accentBright", .6), .9),
    face("rear-window", [[cabinRear, -.56, beltZ], [cabinRear, .56, beltZ], [glassRear, .36, roofZ], [glassRear, -.36, roofZ]], style("accent", .09), style("lineStrong", .42), .9),
    face("roof", [[glassRear, -.36, roofZ], [glassFront, -.33, roofZ], [glassFront, .33, roofZ], [glassRear, .36, roofZ]], style("surface", .84), style("lineStrong", .62), 1),
    face("left-headlamp", [[front - .02, -.5, z + .42], [front - .02, -.18, z + .45], [front - .1, -.2, z + .52], [front - .1, -.52, z + .48]], style("accentBright", .54), style("accentBright", .82), .7),
    face("right-headlamp", [[front - .02, .5, z + .42], [front - .02, .18, z + .45], [front - .1, .2, z + .52], [front - .1, .52, z + .48]], style("accentBright", .5), style("accentBright", .78), .7)
  ];

  const lines = [
    line("left-roof-rail", [[-.42, -.4, railZ], [.44, -.35, railZ]], style("lineStrong", .42), .8),
    line("right-roof-rail", [[-.42, .4, railZ], [.44, .35, railZ]], style("lineStrong", .48), .8),
    line("center-roof-highlight", [[-.3, 0, roofZ + .02], [.32, 0, roofZ + .02]], style("accent", .44), .7),
    line("left-beltline", [[rear + .44, -halfWidth, z + .18], [front - .48, -halfWidth, z + .18]], style("lineStrong", .38), .9),
    line("right-beltline", [[rear + .44, halfWidth, z + .18], [front - .48, halfWidth, z + .18]], style("lineStrong", .44), .9),
    line("rear-fascia", [[rear, -shoulder, deckZ], [rear, shoulder, deckZ]], style("lineStrong", .78), 1.4),
    line("front-fascia", [[front, -.48, deckZ + .03], [front, .48, deckZ + .03]], style("accentBright", .78), 1.1),
    line("windshield-divider", [[-.04, -.4, roofZ + .01], [-.04, .4, roofZ + .01]], style("lineStrong", .24), .7)
  ];

  for (const x of [rearWheelX, frontWheelX]) {
    for (const y of [-wheelY, wheelY]) {
      const outerY = y + Math.sign(y) * .18;
      faces.push(face("wheel", [[x + .38, y, z + .14], [x + .38, outerY, z + .14], [x - .38, outerY, z + .14], [x - .38, y, z + .14]], style("lineStrong", .82), style("surface", .9), .9));
      lines.push(line("wheel-groove", [[x, y, z + .16], [x, outerY, z + .16]], style("surface", .68), .7));
    }
  }

  window.EgoVehicleShape = {
    name: "authored-compact-ego-car",
    source: "Derived from vehicle dimensions only; no confidential mesh data is copied or loaded.",
    dimensions: {
      length: front - rear,
      width,
      wheelBase,
      wheelTread,
      frontOverhang,
      rearOverhang,
      rear,
      front,
      halfWidth
    },
    faces,
    lines,
    axesZ: z + .76
  };
})();
