# Assembly stabilizer pin

Latest fit-first v3 design, dimensions in millimetres: grip diameter 12, grip length 28, shaft diameter 2.3, shaft length 14, total length 42.

- `assembly_pin.scad`: editable OpenSCAD source; render with F6 and export STL.
- `assembly_pin_print_mm.stl`: printable mesh in millimetres.
- `assembly_pin.step`: solid CAD exchange file.

The 14.5 mm shaft in the historical companion source is superseded by this 14 mm version. Fit is based on receiving CAD; physical fit and printer tolerances remain to be checked.

## Original and grouped assembly CAD

- `original-source/`: 10 original STEP files, including the full SO101 assembly, individual printed parts and the wrist-camera mount. Source revision and upstream URLs are retained in `licenses/source-lock.json`; all 10 source hashes were verified before publication.
- `grouped-components-v2/`: 7 STEP exports of the grouped review components: base with servo, side cover, shoulder connector, upper arm, forearm module, camera mount and the historical pin export. The current 14 mm-shaft pin is the root `assembly_pin.*` set; the grouped pin is retained as historical source.
- `licenses/`: upstream Apache-2.0/MIT license texts, attribution and source locks. The STEP source is from TheRobotStudio/SO-ARM100, not a claim of access to a separate WowRobo manufacturer CAD package.
- `CAD_MANIFEST.json`: exact file hashes and provenance.

These are original and grouped source CAD, not a guarantee of manufacturing fit or an editable export of every later native collision/mesh repair. The current inspection meshes remain in `runtime/wowrobo-viewer/meshes/`. No Rhino/native Onshape/Fusion project files were present to publish.

The CAD bundle is also attached to the [states-bank release](https://github.com/nikitamounier/assembly-so101/releases/tag/wowrobo-states-19200-2026-11-07).
