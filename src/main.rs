mod mesh;
mod renderer;
mod scene;

use std::time::Instant;

use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use crate::renderer::Renderer;
use crate::scene::{ORBIT_SAMPLES, Scene};

fn main() -> Result<()> {
    let scene = Scene::create()?;
    scene.export_usda("scene.usda")?;

    let event_loop = EventLoop::new()?;
    let mut app = App {
        scene,
        renderer: None,
        started: Instant::now(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    scene: Scene,
    renderer: Option<Renderer>,
    started: Instant,
}

impl ApplicationHandler for App {
    fn new_events(&mut self, event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }
        if let Err(err) = self.start(event_loop) {
            eprintln!("failed to open the Vulkan window: {err:#}");
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(_) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.note_resize();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(err) = self.draw() {
                    eprintln!("draw failed: {err:#}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = &self.renderer {
            renderer.window().request_redraw();
        }
    }
}

impl App {
    fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("OpenUSD Vulkan")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
        )?;
        self.renderer = Some(Renderer::new(window, self.scene.objects())?);
        self.started = Instant::now();
        Ok(())
    }

    fn draw(&mut self) -> Result<()> {
        let Some(renderer) = &mut self.renderer else {
            return Ok(());
        };
        let turns = self.started.elapsed().as_secs_f64() / 8.0;
        let time = turns * ORBIT_SAMPLES;
        let translations = self.scene.translations(time)?;
        renderer.draw(&translations)
    }
}
