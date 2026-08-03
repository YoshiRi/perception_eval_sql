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
  const shoulder = halfWidth - .08;
  const z = .02;
  const floorZ = z + .18;
  const beltZ = z + .72;
  const windowTopZ = z + 1.7;
  const roofZ = z + 1.88;
  const rearWheelX = -wheelBase / 2;
  const frontWheelX = wheelBase / 2;
  const wheelY = wheelTread / 2;
  const sideInset = halfWidth - .1;
  const windowInset = halfWidth - .2;

  const style = (token, alpha) => ({token, alpha});
  const face = (name, points, fill, stroke, width) => ({name, points, fill, stroke, width});
  const line = (name, points, stroke, width) => ({name, points, stroke, width});

  const faces = [
    face("footprint-shadow", [[rear, -halfWidth, z], [front, -halfWidth, z], [front, halfWidth, z], [rear, halfWidth, z]], style("deep", .1), style("lineStrong", .2), .7),
    face("left-body-side", [[rear, -halfWidth, floorZ], [front, -halfWidth, floorZ], [front, -sideInset, roofZ], [rear, -sideInset, roofZ]], style("surface", .5), style("lineStrong", .52), 1),
    face("right-body-side", [[rear, halfWidth, floorZ], [front, halfWidth, floorZ], [front, sideInset, roofZ], [rear, sideInset, roofZ]], style("surface", .68), style("lineStrong", .58), 1),
    face("front-face", [[front, -halfWidth, floorZ], [front, halfWidth, floorZ], [front, sideInset, roofZ], [front, -sideInset, roofZ]], style("surface", .74), style("accentBright", .64), 1.2),
    face("rear-face", [[rear, -halfWidth, floorZ], [rear, halfWidth, floorZ], [rear, sideInset, roofZ], [rear, -sideInset, roofZ]], style("lineStrong", .1), style("lineStrong", .7), 1.2),
    face("roof", [[rear, -sideInset, roofZ], [front, -sideInset, roofZ], [front, sideInset, roofZ], [rear, sideInset, roofZ]], style("surface", .9), style("lineStrong", .68), 1.1),
    face("left-window-band", [[rear + .42, -windowInset, beltZ], [front - .72, -windowInset, beltZ], [front - .72, -windowInset + .08, windowTopZ], [rear + .42, -windowInset + .08, windowTopZ]], style("deep", .58), style("accent", .62), .9),
    face("right-window-band", [[rear + .42, windowInset, beltZ], [front - .72, windowInset, beltZ], [front - .72, windowInset - .08, windowTopZ], [rear + .42, windowInset - .08, windowTopZ]], style("deep", .66), style("accent", .66), .9),
    face("windshield", [[front, -.58, beltZ], [front, .58, beltZ], [front, .44, windowTopZ], [front, -.44, windowTopZ]], style("accentBright", .2), style("accentBright", .72), .9),
    face("rear-window", [[rear, -.52, beltZ], [rear, .52, beltZ], [rear, .42, windowTopZ], [rear, -.42, windowTopZ]], style("accent", .1), style("lineStrong", .46), .9),
    face("left-lower-panel", [[rear + .28, -halfWidth, floorZ + .08], [front - .28, -halfWidth, floorZ + .08], [front - .28, -halfWidth, beltZ - .12], [rear + .28, -halfWidth, beltZ - .12]], style("surface", .34), style("lineStrong", .34), .7),
    face("right-lower-panel", [[rear + .28, halfWidth, floorZ + .08], [front - .28, halfWidth, floorZ + .08], [front - .28, halfWidth, beltZ - .12], [rear + .28, halfWidth, beltZ - .12]], style("surface", .48), style("lineStrong", .38), .7),
    face("front-route-plate", [[front, -.28, floorZ + .28], [front, .28, floorZ + .28], [front, .28, floorZ + .48], [front, -.28, floorZ + .48]], style("deep", .34), style("accent", .52), .7),
    face("left-headlamp", [[front, -.7, floorZ + .12], [front, -.46, floorZ + .12], [front, -.46, floorZ + .28], [front, -.7, floorZ + .28]], style("accentBright", .58), style("accentBright", .82), .7),
    face("right-headlamp", [[front, .7, floorZ + .12], [front, .46, floorZ + .12], [front, .46, floorZ + .28], [front, .7, floorZ + .28]], style("accentBright", .54), style("accentBright", .78), .7),
    face("left-tail-lamp", [[rear, -.72, floorZ + .18], [rear, -.58, floorZ + .18], [rear, -.58, beltZ - .18], [rear, -.72, beltZ - .18]], style("bad", .38), style("bad", .58), .7),
    face("right-tail-lamp", [[rear, .72, floorZ + .18], [rear, .58, floorZ + .18], [rear, .58, beltZ - .18], [rear, .72, beltZ - .18]], style("bad", .34), style("bad", .54), .7)
  ];

  const lines = [
    line("left-roof-edge", [[rear, -sideInset, roofZ], [front, -sideInset, roofZ]], style("lineStrong", .44), .9),
    line("right-roof-edge", [[rear, sideInset, roofZ], [front, sideInset, roofZ]], style("lineStrong", .5), .9),
    line("left-beltline", [[rear + .2, -halfWidth, beltZ - .08], [front - .2, -halfWidth, beltZ - .08]], style("lineStrong", .38), .9),
    line("right-beltline", [[rear + .2, halfWidth, beltZ - .08], [front - .2, halfWidth, beltZ - .08]], style("lineStrong", .44), .9),
    line("front-bumper", [[front, -shoulder, floorZ + .12], [front, shoulder, floorZ + .12]], style("accentBright", .74), 1.1),
    line("rear-bumper", [[rear, -shoulder, floorZ + .12], [rear, shoulder, floorZ + .12]], style("lineStrong", .66), 1.1),
    line("windshield-center", [[front, 0, beltZ], [front, 0, windowTopZ]], style("lineStrong", .28), .7),
    line("left-window-divider-1", [[rear + 1.18, -windowInset + .03, beltZ], [rear + 1.18, -windowInset + .07, windowTopZ]], style("surface", .52), .7),
    line("left-window-divider-2", [[rear + 2.12, -windowInset + .03, beltZ], [rear + 2.12, -windowInset + .07, windowTopZ]], style("surface", .52), .7),
    line("left-window-divider-3", [[rear + 3.06, -windowInset + .03, beltZ], [rear + 3.06, -windowInset + .07, windowTopZ]], style("surface", .52), .7),
    line("right-window-divider-1", [[rear + 1.18, windowInset - .03, beltZ], [rear + 1.18, windowInset - .07, windowTopZ]], style("surface", .58), .7),
    line("right-window-divider-2", [[rear + 2.12, windowInset - .03, beltZ], [rear + 2.12, windowInset - .07, windowTopZ]], style("surface", .58), .7),
    line("right-window-divider-3", [[rear + 3.06, windowInset - .03, beltZ], [rear + 3.06, windowInset - .07, windowTopZ]], style("surface", .58), .7),
    line("door-cut", [[front - 1.05, halfWidth, floorZ + .08], [front - 1.05, halfWidth, windowTopZ]], style("lineStrong", .5), .8)
  ];

  for (const x of [rearWheelX, frontWheelX]) {
    for (const y of [-wheelY, wheelY]) {
      const outerY = y + Math.sign(y) * .18;
      faces.push(face("wheel", [[x + .34, y, floorZ], [x + .34, outerY, floorZ], [x - .34, outerY, floorZ], [x - .34, y, floorZ]], style("lineStrong", .84), style("surface", .9), .9));
      faces.push(face("wheel-arch", [[x + .48, y, floorZ + .14], [x + .34, y, beltZ - .2], [x - .34, y, beltZ - .2], [x - .48, y, floorZ + .14]], style("deep", .2), style("lineStrong", .44), .7));
      lines.push(line("wheel-groove", [[x, y, floorZ + .02], [x, outerY, floorZ + .02]], style("surface", .68), .7));
    }
  }

  window.EgoVehicleShape = {
    name: "authored-compact-ego-bus",
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
    axesZ: z + .9
  };
})();
