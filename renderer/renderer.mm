// Window host for the WorldOf3D renderer: an NSWindow + MTKView that draws Viewers exactly like
// the app's Renderer (MatrixH.mm drawInMTKView) and routes mouse/keyboard input to them exactly
// like MyMetalView does. The difference is who runs the loop: the app has NSApp.run(); here the
// caller's poll()/show()/wait() pump events and draw, OpenCV-style.
#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>
#include <chrono>
#include "renderer.h"

#ifndef RENDER_METALLIB_PATH
#error "RENDER_METALLIB_PATH must point at the metallib built from Shaders.metal + DAGShaders.metal"
#endif

static constexpr int kSampleCount = 4;
static constexpr double kFrameInterval = 1.0 / 120.0;

static double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Same as MatrixH.mm's ray_from_screen_space, which isn't in a header.
static std::tuple<simd_float3, simd_float3, simd_float2> ray_from_screen_space(simd_float2 view_coords, simd_float2 view_bounds, Camera3D& cam) {
    simd_float2 metal_coords = (2 * view_coords / view_bounds) - 1;
    simd_float4 p1 = simd_make_float4(metal_coords.x, metal_coords.y, 0, 1);
    simd_float4 p2 = simd_make_float4(metal_coords.x, metal_coords.y, 1, 1);
    simd_float4x4 invViewProj = simd_mul(cam.inverseViewMatrix, cam.inverseProjectionMatrix);
    simd_float4 w1 = simd_mul(invViewProj, p1);
    simd_float4 w2 = simd_mul(invViewProj, p2);
    simd_float3 world_p1 = w1.xyz / w1.w;
    simd_float3 world_p2 = w2.xyz / w2.w;
    return {world_p1, world_p2 - world_p1, metal_coords};
}

@class RWView;
@class RWDelegate;

struct Renderer::Impl {
    NSWindow* window = nil;
    RWView* view = nil;
    RWDelegate* delegate = nil;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLRenderPipelineState> states[9] = {};
    id<MTLRenderPipelineState> grid_state = nil;
    id<MTLRenderPipelineState> shadow_state = nil;   // depth-only, for Viewer::prepare's shadow pass
    id<MTLDepthStencilState> depth_state = nil;

    std::vector<std::shared_ptr<Viewer>> figures;   // drawn in order, each in its own viewport
    std::shared_ptr<Viewer> active;                 // figure the current gesture started in
    std::map<Viewer*, float> fit;                   // 2D figures: half-size to keep in view, looking down -z
    bool grid = true;
    MTLClearColor background = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

    bool open = false;
    double last_draw = 0;
    int last_key = -1;
    std::string save_path;   // set by save(): the next frame is also written here
    bool saved = false;

    std::shared_ptr<Viewer> figure_at(NSPoint p, NSSize bounds);
    void add_figure(std::shared_ptr<Viewer> v, simd_float4 rect);
    void draw_frame(MTKView* v);
    void close();
    ~Impl() { close(); }
};

// Every open window, so pumping events from one Renderer keeps them all drawing.
static std::vector<std::weak_ptr<Renderer::Impl>>& windows() {
    static std::vector<std::weak_ptr<Renderer::Impl>> w;
    return w;
}

// ---------------------------------------------------------------------------------------------
// View: input, ported from MyMetalView (MatrixH.mm). Each handler hands the event to the figure
// under the cursor, with a ray from that figure's own camera and viewport, then falls back to the
// same camera call the app makes.
// ---------------------------------------------------------------------------------------------
@interface RWView : MTKView
@property (nonatomic, assign) Renderer::Impl* owner;
@property (nonatomic, strong) NSMutableSet* pressedKeys;
@property (nonatomic, strong) NSTrackingArea* trackingArea;
@property (nonatomic, strong) NSTimer* cameraTimer;
@property (nonatomic, assign) bool shiftPressed;
@end

@implementation RWView

- (instancetype)initWithFrame:(CGRect)frameRect device:(id<MTLDevice>)device {
    self = [super initWithFrame:frameRect device:device];
    if (self) {
        self.pressedKeys = [NSMutableSet set];
        self.cameraTimer = [NSTimer scheduledTimerWithTimeInterval:(1.0 / 120.0) target:self selector:@selector(updateCamera) userInfo:nil repeats:YES];
        self.shiftPressed = false;
    }
    return self;
}

- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)acceptsFirstMouse:(NSEvent*)event { return YES; }

- (void)viewDidMoveToWindow {
    [super viewDidMoveToWindow];
    [self.window setAcceptsMouseMovedEvents:YES];
    [self.window makeFirstResponder:self];
    NSMagnificationGestureRecognizer* mag = [[NSMagnificationGestureRecognizer alloc] initWithTarget:self action:@selector(handleMagnify:)];
    [self addGestureRecognizer:mag];
}

- (void)updateTrackingAreas {
    [super updateTrackingAreas];
    if (self.trackingArea) [self removeTrackingArea:self.trackingArea];
    NSTrackingAreaOptions options = (NSTrackingMouseMoved | NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect);
    self.trackingArea = [[NSTrackingArea alloc] initWithRect:self.bounds options:options owner:self userInfo:nil];
    [self addTrackingArea:self.trackingArea];
}

// The figure's viewport in view coordinates (origin bottom left, like the app's view_coords).
- (NSRect)rectOf:(Viewer*)v {
    NSSize b = self.bounds.size;
    return NSMakeRect(v->viewport.x * b.width, (1.0 - v->viewport.y - v->viewport.w) * b.height,
                      v->viewport.z * b.width, v->viewport.w * b.height);
}

// Ray through `p` (view coordinates) from the figure's camera.
- (std::tuple<simd_float3, simd_float3, simd_float2>)ray:(Viewer*)v at:(NSPoint)p {
    NSRect r = [self rectOf:v];
    return ray_from_screen_space(simd_make_float2(p.x - r.origin.x, p.y - r.origin.y),
                                 simd_make_float2(r.size.width, r.size.height), *v->camera);
}

// `p` in the whole view as -1..1, y up: DrishtiViewerEvent's normalized_pos, which Viewer::in_viewer_space
// maps into the figure's viewport to rebuild the rays from its camera.
- (simd_float2)normalized:(NSPoint)p {
    return 2 * simd_make_float2(p.x / self.bounds.size.width, p.y / self.bounds.size.height) - 1;
}

- (std::shared_ptr<Viewer>)figureAt:(NSPoint)p {
    return self.owner ? self.owner->figure_at(p, self.bounds.size) : nullptr;
}

- (void)mouseMoved:(NSEvent*)event {
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    if (p.x < 0 || p.y < 0 || p.x > self.bounds.size.width || p.y > self.bounds.size.height) return;
    auto v = [self figureAt:p];
    if (!v) return;
    auto [ray_origin, ray_direction, pos] = [self ray:v.get() at:p];
    DrishtiViewerEvent e;
    e.type = DrishtiMouseEvent::Hover;
    e.normalized_pos = e.prev_normalized_pos = [self normalized:p];
    e.zoom_magnitude = 0.0f;
    e.ray_origin = ray_origin;
    e.ray_dir = ray_direction;
    v->handle_event(e);
}

- (void)mouseDown:(NSEvent*)event {
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    auto v = [self figureAt:p];
    self.owner->active = v;
    if (!v) return;
    auto [ray_origin, ray_direction, pos] = [self ray:v.get() at:p];
    DrishtiViewerEvent e;
    e.type = DrishtiMouseEvent::Tap;
    e.normalized_pos = e.prev_normalized_pos = [self normalized:p];
    e.zoom_magnitude = 0.0f;
    e.ray_origin = ray_origin;
    e.ray_dir = ray_direction;
    v->handle_event(e);
}

- (void)rightMouseDown:(NSEvent*)event {
    self.owner->active = [self figureAt:[self convertPoint:event.locationInWindow fromView:nil]];
}

- (void)mouseDragged:(NSEvent*)event {
    float deltaX = event.deltaX;
    float deltaY = event.deltaY;
    BOOL isShift = (event.modifierFlags & NSEventModifierFlagShift) != 0;
    auto v = self.owner->active;
    if (!v) return;
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    auto [ray_origin, ray_direction, pos] = [self ray:v.get() at:p];
    auto [prev_ray_origin, prev_ray_direction, prev_pos] = [self ray:v.get() at:NSMakePoint(p.x - deltaX, p.y + deltaY)];
    DrishtiViewerEvent e;
    e.type = DrishtiMouseEvent::Drag;
    e.normalized_pos = [self normalized:p];
    e.prev_normalized_pos = [self normalized:NSMakePoint(p.x - deltaX, p.y + deltaY)];
    e.delta = simd_make_float2(deltaX, deltaY);
    e.zoom_magnitude = 0.0f;
    e.ray_origin = ray_origin;
    e.ray_dir = ray_direction;
    e.prev_ray_origin = prev_ray_origin;
    e.prev_ray_dir = prev_ray_direction;
    if (v->handle_event(e)) return;
    v->camera->handleMouseEvents(deltaX, -deltaY, NO, NO, isShift ? TransformationMode::Translate : TransformationMode::Orbit);
}

- (void)rightMouseDragged:(NSEvent*)event {
    auto v = self.owner->active;
    if (!v) return;
    v->camera->handleMouseEvents(event.deltaX, event.deltaY, NO, NO, TransformationMode::Translate);
}

- (void)scrollWheel:(NSEvent*)event {
    float deltaY = -event.scrollingDeltaY;
    float deltaX = event.scrollingDeltaX;
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    auto v = [self figureAt:p];
    if (!v) return;
    auto [ray_origin, ray_direction, pos] = [self ray:v.get() at:p];
    auto [prev_ray_origin, prev_ray_direction, prev_pos] = [self ray:v.get() at:NSMakePoint(p.x - deltaX, p.y - deltaY)];
    DrishtiViewerEvent e;
    e.type = DrishtiMouseEvent::Scroll;
    e.normalized_pos = [self normalized:p];
    e.prev_normalized_pos = [self normalized:NSMakePoint(p.x - deltaX, p.y - deltaY)];
    e.delta = simd_make_float2(deltaX, deltaY);
    e.zoom_magnitude = 0.0f;
    e.ray_origin = ray_origin;
    e.ray_dir = ray_direction;
    e.prev_ray_origin = prev_ray_origin;
    e.prev_ray_dir = prev_ray_direction;
    if (v->handle_event(e)) return;
    v->camera->handleMouseEvents(deltaX, deltaY, NO, NO, TransformationMode::Translate);
}

- (void)handleMagnify:(NSMagnificationGestureRecognizer*)gesture {
    float magnification = gesture.magnification;
    NSPoint p = [gesture locationInView:self];
    auto v = [self figureAt:p];
    if (!v) return;
    auto [ray_origin, ray_direction, pos] = [self ray:v.get() at:p];
    DrishtiViewerEvent e;
    e.type = DrishtiMouseEvent::Zoom;
    e.normalized_pos = e.prev_normalized_pos = [self normalized:p];
    e.delta = simd_make_float2(0.0f, 0.0f);
    e.zoom_magnitude = magnification;
    e.ray_origin = ray_origin;
    e.ray_dir = ray_direction;
    e.prev_ray_origin = ray_origin;
    e.prev_ray_dir = ray_direction;
    if (v->handle_event(e)) {
        gesture.magnification = 0.0;
        return;
    }
    v->camera->handleMouseEvents(1 + magnification, 1 + magnification, NO, NO, TransformationMode::Zoom);
    if (gesture.state == NSGestureRecognizerStateEnded) {
        v->camera->updateOLD();
        gesture.magnification = 0.0;
    }
}

- (void)keyDown:(NSEvent*)event {
    [self.pressedKeys addObject:@(event.keyCode)];
    if (event.modifierFlags & NSEventModifierFlagShift) self.shiftPressed = true;
    NSString* chars = event.charactersIgnoringModifiers;
    if (self.owner && chars.length > 0) self.owner->last_key = [chars characterAtIndex:0];
}

- (void)keyUp:(NSEvent*)event {
    [self.pressedKeys removeObject:@(event.keyCode)];
    if (event.modifierFlags & NSEventModifierFlagShift) self.shiftPressed = false;
}

- (void)flagsChanged:(NSEvent*)event {
    if (!(event.modifierFlags & NSEventModifierFlagShift)) self.shiftPressed = false;
}

// Arrow keys / W / S move the camera of the figure last clicked (the first one by default), like the app.
- (void)updateCamera {
    if (!self.owner || self.pressedKeys.count == 0) return;
    float deltaX = 0, deltaY = 0, deltaZ = 0, amount = 20;
    if ([self.pressedKeys containsObject:@123]) deltaX = -amount;   // Left
    if ([self.pressedKeys containsObject:@124]) deltaX = amount;    // Right
    if ([self.pressedKeys containsObject:@125]) deltaY = -amount;   // Down
    if ([self.pressedKeys containsObject:@126]) deltaY = amount;    // Up
    if ([self.pressedKeys containsObject:@13]) deltaZ = amount;     // W
    if ([self.pressedKeys containsObject:@1]) deltaZ = -amount;     // S
    if (deltaX == 0 && deltaY == 0 && deltaZ == 0) return;
    auto v = self.owner->active;
    if (!v && !self.owner->figures.empty()) v = self.owner->figures[0];
    if (!v) return;
    if (self.shiftPressed) {
        v->camera->updateRawCamPosition(simd_make_float3(deltaX * 0.005, deltaY * 0.005, deltaZ * 0.005));
    } else {
        v->camera->handleMouseEvents(deltaX, deltaY, NO, NO, TransformationMode::Translate);
    }
}

- (void)stop {
    [self.cameraTimer invalidate];
    self.cameraTimer = nil;
    self.owner = nullptr;
}
@end

// ---------------------------------------------------------------------------------------------
// Delegate: MTKView draw callback + window close.
// ---------------------------------------------------------------------------------------------
@interface RWDelegate : NSObject <MTKViewDelegate, NSWindowDelegate>
@property (nonatomic, assign) Renderer::Impl* owner;
@end

@implementation RWDelegate
- (void)drawInMTKView:(MTKView*)view {
    if (self.owner) self.owner->draw_frame(view);
}
- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size {}
- (void)windowWillClose:(NSNotification*)notification {
    if (self.owner) self.owner->open = false;
}
@end

@interface RWAppTarget : NSObject
@end
@implementation RWAppTarget
// Cmd+Q / Cmd+W close the windows but never exit(): main() carries on after wait().
- (void)closeAll:(id)sender {
    for (auto& w : windows()) if (auto i = w.lock()) i->close();
}
- (void)closeKey:(id)sender {
    [[NSApp keyWindow] performClose:nil];
}
@end

static void ensure_app() {
    static RWAppTarget* target = nil;
    if (target) return;
    target = [RWAppTarget new];
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    NSMenu* bar = [NSMenu new];
    NSMenuItem* appItem = [NSMenuItem new];
    [bar addItem:appItem];
    NSMenu* appMenu = [NSMenu new];
    NSMenuItem* quit = [[NSMenuItem alloc] initWithTitle:@"Close All Windows" action:@selector(closeAll:) keyEquivalent:@"q"];
    quit.target = target;
    [appMenu addItem:quit];
    NSMenuItem* closeItem = [[NSMenuItem alloc] initWithTitle:@"Close Window" action:@selector(closeKey:) keyEquivalent:@"w"];
    closeItem.target = target;
    [appMenu addItem:closeItem];
    appItem.submenu = appMenu;
    NSApp.mainMenu = bar;

    [NSApp finishLaunching];
    [NSApp activateIgnoringOtherApps:YES];
}

// Handles every queued event, timer and main-queue block (the controllers' dispatch_async updates).
// Waits up to `timeout` seconds for the first event.
static void pump(double timeout) {
    @autoreleasepool {
        NSDate* until = timeout > 0 ? [NSDate dateWithTimeIntervalSinceNow:timeout] : [NSDate distantPast];
        while (NSEvent* e = [NSApp nextEventMatchingMask:NSEventMaskAny untilDate:until inMode:NSDefaultRunLoopMode dequeue:YES]) {
            [NSApp sendEvent:e];
            until = [NSDate distantPast];
        }
        while (CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, true) == kCFRunLoopRunHandledSource) {}
    }
}

static bool any_open() {
    for (auto& w : windows()) if (auto i = w.lock(); i && i->open) return true;
    return false;
}

// Draws every open window whose next frame is due (or all of them, `force`).
static void draw_windows(bool force) {
    double t = now_seconds();
    for (auto& w : windows()) {
        auto i = w.lock();
        if (!i || !i->open) continue;
        if (!force && t - i->last_draw < kFrameInterval) continue;
        i->last_draw = t;
        @autoreleasepool { [i->view draw]; }
    }
}

static bool write_png(id<MTLTexture> tex, const std::string& path) {
    size_t w = tex.width, h = tex.height, row = w * 4;
    std::vector<uint8_t> px(row * h);
    [tex getBytes:px.data() bytesPerRow:row fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    // The window is opaque: ignore the framebuffer alpha (blending leaves it < 1 under translucent
    // geometry, and treating it as premultiplied would divide the colour by it).
    CGContextRef ctx = CGBitmapContextCreate(px.data(), w, h, 8, row, cs, kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
    CGImageRef img = CGBitmapContextCreateImage(ctx);
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    CGImageDestinationRef dst = CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
    bool ok = false;
    if (dst) {
        CGImageDestinationAddImage(dst, img, nullptr);
        ok = CGImageDestinationFinalize(dst);
        CFRelease(dst);
    }
    CGImageRelease(img);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    return ok;
}

// ---------------------------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------------------------
std::shared_ptr<Viewer> Renderer::Impl::figure_at(NSPoint p, NSSize b) {
    for (auto it = figures.rbegin(); it != figures.rend(); ++it) {
        auto& v = *it;
        if (!v->visible) continue;
        double x0 = v->viewport.x * b.width, x1 = x0 + v->viewport.z * b.width;
        double y1 = (1.0 - v->viewport.y) * b.height, y0 = y1 - v->viewport.w * b.height;
        if (p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1) return v;
    }
    return nullptr;
}

// Puts a viewer in `rect` with a camera of its own: 2D viewers are framed head-on (and re-framed
// when their tile changes shape, see draw_frame), anything else looks down on the ground grid.
void Renderer::Impl::add_figure(std::shared_ptr<Viewer> v, simd_float4 rect) {
    v->viewport = rect;
    if (dynamic_cast<GraphViewer*>(v.get())) fit[v.get()] = 1.08f;          // the graph frame spans [-1, 1]
    else if (dynamic_cast<PlotterViewer*>(v.get())) fit[v.get()] = 0.7f;    // plot quad [-0.5, 0.5] + the scale bar at x = -0.6
    if (!v->camera) {
        v->camera = std::make_shared<Camera3D>();
        if (!fit.count(v.get())) v->camera->lookAt(simd_make_float3(1.8f, 1.4f, -2.4f), simd_make_float3(0, 0, 0));
    }
    figures.push_back(std::move(v));
}

// The app's drawInMTKView, minus the legacy queues: each figure, with the ground grid behind it if it has show_grid,
// in its viewport with its camera.
void Renderer::Impl::draw_frame(MTKView* view) {
    MTLRenderPassDescriptor* pass = [view currentRenderPassDescriptor];
    id<CAMetalDrawable> drawable = [view currentDrawable];
    if (!pass || !drawable) return;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = background;
    pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
    pass.depthAttachment.clearDepth = 1.0;

    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    // Passes that must run before the main one (shadow maps).
    for (auto& v : figures) if (v->visible) v->prepare(cmdBuffer, device, shadow_state);
    id<MTLRenderCommandEncoder> enc = [cmdBuffer renderCommandEncoderWithDescriptor:pass];
    [enc setDepthStencilState:depth_state];
    [enc setCullMode:MTLCullModeNone];
    [enc setFrontFacingWinding:MTLWindingCounterClockwise];

    auto draw_grid = [&](Camera3D& c) {
        [enc setRenderPipelineState:grid_state];
        [enc setVertexBytes:&c.viewMatrix length:sizeof(simd_float4x4) atIndex:0];
        [enc setVertexBytes:&c.target length:sizeof(simd_float3) atIndex:1];
        [enc setVertexBytes:&c.inverseProjectionMatrix length:sizeof(simd_float4x4) atIndex:2];
        [enc setFragmentBytes:&c.target length:sizeof(simd_float3) atIndex:0];
        [enc setFragmentBytes:&c.inverseProjectionMatrix length:sizeof(simd_float4x4) atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    };

    const double target_w = pass.colorAttachments[0].texture.width;
    const double target_h = pass.colorAttachments[0].texture.height;
    auto in_flight = std::make_shared<std::vector<matrix>>();
    int active_state = -1;
    for (auto& v : figures) {
        if (!v->visible) continue;
        double x = v->viewport.x * target_w, y = v->viewport.y * target_h;
        double w = v->viewport.z * target_w, h = v->viewport.w * target_h;
        long x0 = std::lround(std::max(x, 0.0)), x1 = std::lround(std::min(x + w, target_w));
        long y0 = std::lround(std::max(y, 0.0)), y1 = std::lround(std::min(y + h, target_h));
        if (x1 <= x0 || y1 <= y0) continue;
        [enc setViewport:(MTLViewport){x, y, w, h, 0.0, 1.0}];
        [enc setScissorRect:(MTLScissorRect){(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)}];

        Camera3D* c = v->camera.get();
        float aspect = (float)(w / h);
        if (c->getAspectRatio() != aspect) {
            c->updateAspectRatio(aspect);
            // Re-frame 2D figures so the whole plot fits a tall or wide tile (fov is vertical).
            if (auto f = fit.find(v.get()); f != fit.end()) {
                float d = f->second / std::tan((float)M_PI * 0.125f) / std::min(1.0f, aspect);
                c->lookAt(simd_make_float3(0, 0, -d), simd_make_float3(0, 0, 0));
            }
        }
        c->updateMatrix();
        if (grid && v->show_grid) {
            draw_grid(*c);
            active_state = -1;
        }
        [enc setVertexBytes:&c->viewMatrix length:sizeof(simd_float4x4) atIndex:2];
        v->draw(enc, device, states, c, active_state);
        std::move(v->frame_refs.begin(), v->frame_refs.end(), std::back_inserter(*in_flight));
        v->frame_refs.clear();
    }

    [enc endEncoding];
    [cmdBuffer presentDrawable:drawable];
    // Released on main, where the controllers swap node data (drained by the next pump()).
    [cmdBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        dispatch_async(dispatch_get_main_queue(), ^{ in_flight->clear(); });
    }];
    [cmdBuffer commit];

    if (!save_path.empty()) {
        [cmdBuffer waitUntilCompleted];
        saved = write_png(drawable.texture, save_path);
        save_path.clear();
    }
}

void Renderer::Impl::close() {
    if (!window) return;
    open = false;
    [view stop];
    view.delegate = nil;
    delegate.owner = nullptr;
    window.delegate = nil;
    [window close];
    window = nil;
    view = nil;
    delegate = nil;
}

// ---------------------------------------------------------------------------------------------
// Pipelines: the app's slots 3-8 (Renderer initWithDevice, MatrixH.mm) + the infinite grid.
// ---------------------------------------------------------------------------------------------
static id<MTLRenderPipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> lib, NSString* vs, NSString* fs, bool blend, bool full_blend) {
    id<MTLFunction> vfn = [lib newFunctionWithName:vs];
    id<MTLFunction> ffn = [lib newFunctionWithName:fs];
    if (!vfn || !ffn) {
        NSLog(@"renderer: shader %@ / %@ not found", vs, fs);
        return nil;
    }
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    if (blend) {
        d.colorAttachments[0].blendingEnabled = YES;
        d.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        d.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        if (full_blend) {
            d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
            d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        }
    }
    d.vertexDescriptor = nil;
    d.vertexFunction = vfn;
    d.fragmentFunction = ffn;
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth16Unorm;
    d.rasterSampleCount = kSampleCount;
    NSError* error = nil;
    id<MTLRenderPipelineState> s = [device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!s) NSLog(@"renderer: pipeline %@ failed: %@", vs, error);
    return s;
}

// The shadow pass (Viewer::prepare): depth only, into a single-sample Depth32Float shadow map.
static id<MTLRenderPipelineState> make_shadow_pipeline(id<MTLDevice> device, id<MTLLibrary> lib) {
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:@"shadow_vertex"];
    d.fragmentFunction = nil;                                  // depth only
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    d.rasterSampleCount = 1;
    NSError* error = nil;
    id<MTLRenderPipelineState> s = [device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!s) NSLog(@"renderer: shadow pipeline failed: %@", error);
    return s;
}

// ---------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------
Renderer Renderer::init(int width, int height, const std::string& title) {
    ensure_app();
    Renderer r;
    r.impl = std::make_shared<Impl>();
    Impl& i = *r.impl;

    i.device = MTLCreateSystemDefaultDevice();
    i.queue = [i.device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> lib = [i.device newLibraryWithURL:[NSURL fileURLWithPath:@RENDER_METALLIB_PATH] error:&error];
    if (!lib) {
        NSLog(@"renderer: couldn't load %s: %@", RENDER_METALLIB_PATH, error);
        std::abort();
    }
    i.states[3] = make_pipeline(i.device, lib, @"DAGNodeVertexShader", @"DAGNodeFragmentShader", true, false);
    i.states[4] = make_pipeline(i.device, lib, @"vertex_line3d", @"fragment_line3d", true, false);
    i.states[5] = make_pipeline(i.device, lib, @"vertex_dag_pointcloud", @"fragment_dag_pointcloud", true, false);
    i.states[6] = make_pipeline(i.device, lib, @"vertex_edge3d", @"fragment_edge3d", true, false);
    i.states[7] = make_pipeline(i.device, lib, @"vertex_label", @"fragment_label", true, false);
    i.states[8] = make_pipeline(i.device, lib, @"vertex_line3d_round", @"fragment_line3d", true, false);
    i.grid_state = make_pipeline(i.device, lib, @"InfiniteGridVertexShader", @"InfiniteGridFragmentShader", true, true);
    i.shadow_state = make_shadow_pipeline(i.device, lib);
    MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
    ds.depthCompareFunction = MTLCompareFunctionLess;
    ds.depthWriteEnabled = YES;
    i.depth_state = [i.device newDepthStencilStateWithDescriptor:ds];

    NSRect frame = NSMakeRect(0, 0, width, height);
    i.window = [[NSWindow alloc] initWithContentRect:frame
                                           styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable
                                             backing:NSBackingStoreBuffered
                                               defer:NO];
    i.window.releasedWhenClosed = NO;
    i.window.title = [NSString stringWithUTF8String:title.c_str()];

    i.view = [[RWView alloc] initWithFrame:frame device:i.device];
    i.view.owner = &i;
    i.view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    i.view.depthStencilPixelFormat = MTLPixelFormatDepth16Unorm;
    i.view.sampleCount = kSampleCount;
    i.view.clearDepth = 1.0;
    i.view.framebufferOnly = NO;
    i.view.paused = YES;                 // we draw from poll()/show()/wait(), not a display link
    i.view.enableSetNeedsDisplay = NO;

    i.delegate = [RWDelegate new];
    i.delegate.owner = &i;
    i.view.delegate = i.delegate;
    i.window.delegate = i.delegate;
    i.window.contentView = i.view;

    [i.window center];
    [i.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    i.open = true;
    windows().push_back(r.impl);
    pump(0);
    return r;
}

void Renderer::layout(const std::vector<std::shared_ptr<Viewer>>& viewers, int rows, int cols) {
    if (rows < 1 || cols < 1 || (size_t)(rows * cols) < viewers.size()) {
        NSLog(@"renderer: subplots(%d, %d) has no room for %zu viewers", rows, cols, viewers.size());
        std::abort();
    }
    clear();
    for (size_t k = 0; k < viewers.size(); k++) {
        float r = (float)(k / cols), c = (float)(k % cols);
        impl->add_figure(viewers[k], simd_make_float4(c / cols, r / rows, 1.0f / cols, 1.0f / rows));
    }
}

void Renderer::add_viewer(std::shared_ptr<Viewer> viewer, simd_float4 rect) { impl->add_figure(std::move(viewer), rect); }

void Renderer::clear() {
    impl->figures.clear();
    impl->fit.clear();
    impl->active.reset();
}

void Renderer::set_grid(bool on) { impl->grid = on; }
void Renderer::set_background(simd_float4 c) { impl->background = MTLClearColorMake(c.x, c.y, c.z, c.w); }

static int take_key() {
    int key = -1;
    for (auto& w : windows()) {
        if (auto i = w.lock(); i && i->last_key != -1) {
            key = i->last_key;
            i->last_key = -1;
        }
    }
    return key;
}

int Renderer::poll() {
    pump(0);
    draw_windows(false);
    return take_key();
}

int Renderer::show() {
    pump(0);
    draw_windows(true);
    pump(0);
    return take_key();
}

int Renderer::wait_key(int ms) {
    double deadline = ms > 0 ? now_seconds() + ms / 1000.0 : INFINITY;
    while (any_open()) {
        pump(kFrameInterval);
        draw_windows(false);
        if (int key = take_key(); key != -1) return key;
        if (now_seconds() >= deadline) break;
    }
    return -1;
}

void Renderer::wait() {
    while (any_open()) {
        pump(kFrameInterval);
        draw_windows(false);
    }
    take_key();
}

bool Renderer::save(const std::string& png_path) {
    if (!is_open()) return false;
    pump(0);
    impl->save_path = png_path;
    impl->saved = false;
    impl->last_draw = now_seconds();
    @autoreleasepool { [impl->view draw]; }
    impl->save_path.clear();
    return impl->saved;
}

bool Renderer::is_open() const { return impl && impl->open; }

void Renderer::close() {
    if (impl) impl->close();
    pump(0);
}
