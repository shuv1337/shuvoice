use shuvoice_app::media::MediaTransport;
use shuvoice_core::Config;
use std::sync::Arc;

pub fn transport(config: &Config) -> Option<Arc<dyn MediaTransport>> {
    if !config.pause_media_during_recording {
        return None;
    }
    #[cfg(feature = "media-control")]
    {
        Some(Arc::new(Native(shuvoice_io::media::MprisClient::new())))
    }
    #[cfg(not(feature = "media-control"))]
    {
        None
    }
}

#[cfg(feature = "media-control")]
struct Native(shuvoice_io::media::MprisClient);
#[cfg(feature = "media-control")]
#[async_trait::async_trait]
impl MediaTransport for Native {
    async fn shutdown(&self) {
        self.0.shutdown().await;
    }
    async fn subscribe(
        &self,
    ) -> Result<shuvoice_app::media::MediaEventReceiver, shuvoice_core::media::MediaError> {
        self.0.subscribe().await
    }
    async fn discover(
        &self,
    ) -> Result<Vec<shuvoice_core::media::MediaEndpoint>, shuvoice_core::media::MediaError> {
        self.0.discover().await
    }
    async fn snapshot(
        &self,
        e: &shuvoice_core::media::MediaEndpoint,
    ) -> Result<shuvoice_core::media::MediaSnapshot, shuvoice_core::media::MediaError> {
        self.0.snapshot(e).await
    }
    async fn owner_matches(
        &self,
        e: &shuvoice_core::media::MediaEndpoint,
    ) -> Result<bool, shuvoice_core::media::MediaError> {
        self.0.owner_matches(e).await
    }
    async fn pause(
        &self,
        e: &shuvoice_core::media::MediaEndpoint,
    ) -> shuvoice_core::media::MediaCallOutcome {
        self.0.pause(e).await
    }
    async fn play(
        &self,
        e: &shuvoice_core::media::MediaEndpoint,
    ) -> shuvoice_core::media::MediaCallOutcome {
        self.0.play(e).await
    }
}
