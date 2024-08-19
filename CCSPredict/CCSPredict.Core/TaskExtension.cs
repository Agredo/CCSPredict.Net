namespace CCSPredict.Core;

public static class TaskExtension
{
    public static void Await(this Task task)
    {
        task.Wait();
    }
}
