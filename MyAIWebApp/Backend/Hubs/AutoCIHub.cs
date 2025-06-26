using Microsoft.AspNetCore.SignalR;

namespace MyAIWebApp.Backend
{
    public class AutoCIHub : Hub
    {
        private readonly ILogger<AutoCIHub> _logger;

        public AutoCIHub(ILogger<AutoCIHub> logger)
        {
            _logger = logger;
        }

        public override async Task OnConnectedAsync()
        {
            _logger.LogInformation($"Client connected: {Context.ConnectionId}");
            await Clients.Caller.SendAsync("Connected", new { connectionId = Context.ConnectionId });
            await base.OnConnectedAsync();
        }

        public override async Task OnDisconnectedAsync(Exception? exception)
        {
            _logger.LogInformation($"Client disconnected: {Context.ConnectionId}");
            await base.OnDisconnectedAsync(exception);
        }

        public async Task SubscribeToProcess(string processId)
        {
            await Groups.AddToGroupAsync(Context.ConnectionId, $"process-{processId}");
            await Clients.Caller.SendAsync("SubscribedToProcess", processId);
        }

        public async Task UnsubscribeFromProcess(string processId)
        {
            await Groups.RemoveFromGroupAsync(Context.ConnectionId, $"process-{processId}");
            await Clients.Caller.SendAsync("UnsubscribedFromProcess", processId);
        }
    }
}